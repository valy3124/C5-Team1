"""
fine_tune_detr.py — Fine-tuning script for DETR on KITTI-MOTS.

This script is intentionally scoped to **HuggingFace DETR only**.
Torchvision Faster R-CNN and Ultralytics YOLO use different training APIs and should
be fine-tuned with their own dedicated scripts:

    fine_tune.py       – Torchvision Faster R-CNN
    fine_tune_yolo.py  – Ultralytics YOLO

All three scripts share the same data pipeline (KITTI-MOTS dataset wrappers,
Albumentations augmentation, COCO evaluation) so that results are comparable.

Pipeline overview (this script):
    1. setup_experiment  – load YAML config, init W&B, create output directory.
    2. setup_data        – build KITTI-MOTS datasets + DataLoaders.
    3. setup_model       – instantiate DETR, ImageProcessor, optimizer, and scheduler.
    4. train             – run the training loop with periodic COCO evaluation.
"""

# ---------------------------------------------------------------------------
# 1. Path bootstrap
# ---------------------------------------------------------------------------
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))  # …/fine_tune/
src_dir     = os.path.dirname(current_dir)                # …/src/
week1_dir   = os.path.dirname(src_dir)                    # …/Week1/

for p in (src_dir, week1_dir):
    if p not in sys.path:
        sys.path.append(p)

# ---------------------------------------------------------------------------
# 2. Standard library & third-party imports
# ---------------------------------------------------------------------------
import argparse
import json

import albumentations as A
import numpy as np
import torch
import torch.optim as optim
import wandb
import yaml
from albumentations.pytorch import ToTensorV2
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from transformers import DetrForObjectDetection, DetrImageProcessor

# ---------------------------------------------------------------------------
# 3. Project imports
# ---------------------------------------------------------------------------
import datasets
from inference.evaluation import CocoMetrics


# ===========================================================================
# Dataset Adapters
# ===========================================================================

class KITTIMOTSToTorchvision(torch.utils.data.Dataset):
    """Adapter that converts a raw KITTI-MOTS dataset into a format suitable
    for Albumentations and later HuggingFace Processors.

    The raw dataset yields ``(PIL.Image, [annotation], metadata)`` tuples.
    This adapter converts them to ``(np.ndarray image, target dict)`` tuples.

    Target dict schema
    ------------------
    boxes    : FloatTensor[N, 4]  – bounding boxes in *xyxy* format (x2, y2 exclusive).
    labels   : Int64Tensor[N]     – class indices (1 = Car, 2 = Pedestrian).
    image_id : Int64Tensor[1]     – dataset index used as a unique image identifier.
    area     : FloatTensor[N]     – box areas in pixels².
    iscrowd  : Int64Tensor[N]     – all zeros (KITTI-MOTS has no crowd annotations).
    """

    def __init__(self, base_ds: torch.utils.data.Dataset) -> None:
        self.base_ds = base_ds

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        img_pil, raw_anns, _ = self.base_ds[idx]
        img_np = np.array(img_pil)
        target = self._build_target(raw_anns, image_id=idx)
        return img_np, target

    def _build_target(self, raw_anns: List[Any], image_id: int) -> Dict[str, torch.Tensor]:
        """Convert raw annotation objects into a compatible target dict."""
        boxes, labels = [], []

        for ann in raw_anns:
            cls = int(getattr(ann, "class_id", -1))
            box = getattr(ann, "bbox_xyxy", None)
            if box is None:
                continue

            x1, y1, x2, y2 = map(float, box)

            # KITTI-MOTS uses *inclusive* pixel indices for the max corner.
            # Add 1 to make x2/y2 exclusive so that (x2 - x1) gives the correct width.
            x2 += 1
            y2 += 1

            if x2 <= x1 or y2 <= y1:
                continue  # skip degenerate boxes

            boxes.append([x1, y1, x2, y2])
            labels.append(cls)

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        # Area is derived from boxes (not copied from annotation) for consistency.
        area = (
            (boxes_t[:, 2] - boxes_t[:, 0]).clamp(min=0) *
            (boxes_t[:, 3] - boxes_t[:, 1]).clamp(min=0)
        ) if boxes_t.numel() else torch.zeros((0,), dtype=torch.float32)

        return {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area":     area.to(torch.float32),
            "iscrowd":  torch.zeros((labels_t.shape[0],), dtype=torch.int64),
        }


class ApplyAlbumentations(torch.utils.data.Dataset):
    """Wrapper that applies an Albumentations pipeline (including bounding-box
    augmentation) to each sample at load time.

    Args:
        ds:          Source dataset returning ``(np.ndarray, target_dict)`` pairs.
        tf:          An ``A.Compose`` transform with ``BboxParams`` configured.
                     Pass ``None`` to skip augmentation (image is converted to
                     a float tensor via ``to_tensor``).
        keep_empty:  If ``False``, samples whose boxes are completely removed by
                     clipping/visibility filtering are skipped by cycling to the
                     next index.  Defaults to ``True``.
    """

    def __init__(
        self,
        ds: torch.utils.data.Dataset,
        tf: Optional[A.Compose],
        keep_empty: bool = True,
    ) -> None:
        self.ds         = ds
        self.tf         = tf
        self.keep_empty = keep_empty

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        img_np, target = self.ds[idx]

        if self.tf is not None:
            boxes  = target["boxes"].numpy().tolist()
            labels = target["labels"].numpy().tolist()

            out      = self.tf(image=img_np, bboxes=boxes, class_labels=labels)
            img      = out["image"]
            boxes_tf = out["bboxes"]
            labels_tf = out["class_labels"]

            # Ensure the image is a float tensor in [0, 1].
            if isinstance(img, torch.Tensor):
                img = img.float().div(255.0) if img.dtype == torch.uint8 else img.float()
            else:
                img = to_tensor(img)  # fallback: PIL/ndarray → float [0, 1]
        else:
            img       = to_tensor(img_np)
            boxes_tf  = target["boxes"].numpy().tolist()
            labels_tf = target["labels"].numpy().tolist()

        # When all boxes are clipped away and keep_empty is False, cycle to
        # the next sample to avoid feeding empty targets to the model.
        if len(boxes_tf) == 0 and not self.keep_empty:
            return self.__getitem__((idx + 1) % len(self))

        if boxes_tf:
            target["boxes"]  = torch.tensor(boxes_tf,  dtype=torch.float32)
            target["labels"] = torch.tensor(labels_tf, dtype=torch.int64)
        else:
            target["boxes"]  = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,),   dtype=torch.int64)

        # Recompute area from the (possibly augmented) boxes.
        target["area"] = (
            (target["boxes"][:, 2] - target["boxes"][:, 0]).clamp(min=0) *
            (target["boxes"][:, 3] - target["boxes"][:, 1]).clamp(min=0)
        ) if target["boxes"].numel() else torch.zeros((0,), dtype=torch.float32)

        return img, target


# ===========================================================================
# Augmentation Strategies
# ===========================================================================

def get_transforms(is_train: bool = True, aug_strategy: str = "base") -> A.Compose:
    """Build an Albumentations pipeline for training or validation.

    All pipelines keep bounding boxes in *pascal_voc* (xyxy) format and
    automatically clip boxes to image boundaries and remove those with
    area < 1 px² or visibility < 10 %.

    Training augmentation strategies
    ---------------------------------
    ``base``             – Horizontal flip only (safe default).
    ``geometric``        – base + ShiftScaleRotate + Perspective.
    ``color_jitter``     – base + ColorJitter + HueSaturation + Gamma + RGBShift.
    ``extreme_weather``  – base + rain / fog / shadow / sun-flare (p=0.8).
    ``heavy_corruption`` – base + GaussNoise + MotionBlur + CoarseDropout + JPEG compression.
    ``limit_test``       – All of the above combined (useful for ablations).
    ``legacy``           – Original strategy used in early experiments.

    Validation pipeline always returns a plain tensor with no geometric changes.

    Args:
        is_train:     Whether to build the training pipeline.
        aug_strategy: Name of the augmentation strategy (ignored for val).

    Returns:
        An ``A.Compose`` instance ready for use in ``ApplyAlbumentations``.
    """
    bbox_params = A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        clip=True,
        min_area=1,
        min_visibility=0.1,
    )

    if not is_train:
        # Validation: convert to tensor only, no geometry changes.
        return A.Compose(
            [ToTensorV2()],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                clip=True,
            ),
        )

    # ----- Build the list of train transforms -----

    if aug_strategy == "legacy":
        # Reproduces the very first set of experiments (kept for backwards
        # compatibility and comparison with older checkpoints).
        tfms = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, border_mode=0, p=0.5),
            A.OneOf([
                A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1),
                A.RandomFog(fog_coef_range=(0.3, 1), alpha_coef=0.08, p=1),
                A.RandomShadow(p=1),
            ], p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(p=0.3),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, p=0.3),
            A.MotionBlur(p=0.2),
        ]
    else:
        # Every non-legacy strategy starts from a horizontal flip.
        tfms = [A.HorizontalFlip(p=0.5)]

        if aug_strategy in ("geometric", "limit_test"):
            tfms += [
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15, border_mode=0, p=0.5),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
            ]

        if aug_strategy in ("color_jitter", "limit_test"):
            tfms += [
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.HueSaturationValue(p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.RGBShift(p=0.2),
            ]

        if aug_strategy in ("extreme_weather", "limit_test"):
            prob = 0.5 if aug_strategy == "limit_test" else 0.8
            tfms += [
                A.OneOf([
                    A.RandomRain(brightness_coefficient=0.7, drop_width=1, blur_value=5, p=1),
                    A.RandomFog(fog_coef_range=(0.4, 0.9), alpha_coef=0.1, p=1),
                    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, p=1),
                    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, p=1),
                ], p=prob),
            ]

        if aug_strategy in ("heavy_corruption", "limit_test"):
            tfms += [
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.MotionBlur(blur_limit=(3, 7), p=0.3),
                A.CoarseDropout(num_holes_range=(2, 10), hole_height_range=(10, 40), hole_width_range=(10, 40), fill=0, p=0.4),
                A.ImageCompression(quality_lower=50, quality_upper=90, p=0.3),
            ]

    # ToTensorV2 must always be the last transform.
    tfms.append(ToTensorV2())
    return A.Compose(tfms, bbox_params=bbox_params)


# ===========================================================================
# Data containers (plain dataclasses — no logic, just structured namespaces)
# ===========================================================================

@dataclass
class Exp:
    """Experiment-level configuration and environment state."""
    cfg:             Dict[str, Any]  # Full parsed YAML configuration
    device:          Any             # torch.device for training/inference
    output_dir:      str             # Directory where artifacts are saved
    best_model_path: str             # Full path to the best checkpoint file


@dataclass
class Data:
    """Dataset objects and DataLoaders for a single experiment."""
    classes:           List[str]  # Ordered list of class names (index 0 = background)
    train_loader:      Any        # DataLoader for the training split
    val_loader:        Any        # DataLoader for the validation split
    train_coco_metrics: Any = None  # CocoMetrics helper for the train split
    val_coco_metrics:   Any = None  # CocoMetrics helper for the val split


@dataclass
class Run:
    """Model, optimizer, scheduler, and live training state."""
    model:      Any               # Torch model instance
    processor:  Any               # HF DetrImageProcessor
    optimizer:  Any               # Optimizer instance (AdamW, SGD, …)
    history:    Dict[str, list]   # Tracked metrics per epoch
    scheduler:  Any  = None       # Optional LR scheduler
    best_map:   float = 0.0       # Best COCO AP seen so far
    best_epoch: int   = 0         # Epoch at which best_map was achieved


@dataclass
class Eval:
    """Outputs produced by a single evaluation pass."""
    predictions: List[Dict]      # COCO-format prediction dicts
    metrics:     Dict[str, float]  # Metric name → value (e.g. "overall/AP")
    coco_eval:   Any = None       # Raw COCOeval object (optional, for debugging)


# ===========================================================================
# Utility helpers
# ===========================================================================

def collate_fn(batch):
    """Custom collate that keeps images and targets as plain Python lists.

    HF DETR processor will handle the batching and padding dynamically
    during the training loop.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def _xyxy_to_xywh(box: List[float]) -> List[float]:
    """Convert a bounding box from xyxy to xywh format (COCO standard).

    Args:
        box: [x1, y1, x2, y2] coordinates.

    Returns:
        [x, y, width, height]
    """
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


# ===========================================================================
# Experiment setup
# ===========================================================================

def setup_experiment(config_path: str, args: argparse.Namespace) -> Exp:
    """Load config, apply CLI overrides, initialise W&B, and set up I/O paths."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    # --- Apply CLI overrides ---
    training = cfg["training"]

    if args.epochs:     training["epochs"]     = args.epochs
    if args.batch_size: training["batch_size"] = args.batch_size
    if args.lr:         training["lr"]         = args.lr
    if args.t_max:
        training.setdefault("scheduler", {})["t_max"] = args.t_max

    if args.project: cfg["project"]         = args.project
    if args.name:    cfg["experiment_name"] = args.name

    if args.freeze_strategy  is not None: training["freeze_strategy"]  = int(args.freeze_strategy)
    if args.aug_strategy     is not None: training["aug_strategy"]     = str(args.aug_strategy)
    if args.name_fields      is not None: training["name_fields"]      = str(args.name_fields)

    # Data split settings (seeded sub-splits for hyper-parameter search)
    cfg.setdefault("data", {})
    cfg["data"]["mode"]        = args.mode        or cfg["data"].get("mode",        "full")
    cfg["data"]["seed"]        = args.seed        or cfg["data"].get("seed",        42)
    cfg["data"]["split_ratio"] = args.split_ratio or cfg["data"].get("split_ratio", 0.8)

    # --- Auto-generate experiment name from config fields if not given explicitly ---
    if not args.name:
        name_fields_str = training.get("name_fields", "freeze_strategy,lr")
        parts = [cfg["model"]["name"]]
        for field in (f.strip() for f in name_fields_str.split(",") if f.strip()):
            val = training.get(field, "Unknown")
            label = {
                "aug_strategy":    f"Aug_{val}",
                "freeze_strategy": f"Freeze_L{val}",
                "lr":              f"LR_{val}",
            }.get(field, f"{field}_{val}")
            parts.append(label)
        cfg["experiment_name"] = "_".join(parts)
        print(f"Auto-generated experiment name: {cfg['experiment_name']}")

    # --- W&B initialisation ---
    wandb.init(
        project=cfg.get("project", "kitti-mots-finetune"),
        name=cfg.get("experiment_name", "run"),
        config=cfg,
    )

    # Place results under <output_dir>/<model_name>/<experiment_name>_<wandb_id>/
    model_dir  = cfg["model"].get("name", "unknown_model")
    output_dir = os.path.join(
        cfg.get("output_dir", "results"),
        model_dir,
        f"{cfg.get('experiment_name', 'exp')}_{wandb.run.id}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Persist the effective configuration for reproducibility.
    with open(os.path.join(output_dir, "config.yaml"), "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    best_model_path = os.path.join(output_dir, "best_model.pth")
    return Exp(cfg, device, output_dir, best_model_path)


def build_scheduler(optimizer, cfg: Dict[str, Any]) -> Optional[Any]:
    """Construct an LR scheduler from the ``training.scheduler`` config block."""
    sch_cfg = cfg["training"].get("scheduler")
    if sch_cfg is None:
        return None

    name = sch_cfg.get("name", "none").lower()

    if name in ("none", "null"):
        return None

    if name in ("step", "steplr"):
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sch_cfg.get("step_size", 3),
            gamma=sch_cfg.get("gamma", 0.1),
        )

    if name in ("cosine", "cosineannealing", "cosineannealinglr"):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sch_cfg.get("t_max", cfg["training"].get("epochs", 10)),
            eta_min=sch_cfg.get("eta_min", 0.0),
        )

    raise ValueError(f"Unknown scheduler '{name}'.  Valid options: none, step, cosine.")


def setup_data(exp: Exp) -> Data:
    """Instantiate KITTI-MOTS datasets and wrap them with DataLoaders."""
    cfg          = exp.cfg
    root         = cfg["data"]["root"]
    mode         = cfg["data"].get("mode",         "full")
    seed         = cfg["data"].get("seed",         42)
    split_ratio  = cfg["data"].get("split_ratio",  0.8)
    aug_strategy = cfg["training"].get("aug_strategy", "legacy")

    if mode == "full":
        train_split, val_split = "train_full", "validation"
    elif mode == "search":
        train_split, val_split = "train", "dev"
    else:
        raise ValueError(f"Unknown data.mode '{mode}'.  Valid options: full, search.")

    # Raw datasets → torchvision-compatible adapter → augmentation wrapper
    train_ds = ApplyAlbumentations(
        ds=KITTIMOTSToTorchvision(
            datasets.KITTIMOTS(root, split=train_split, ann_source="txt", seed=seed, split_ratio=split_ratio)
        ),
        tf=get_transforms(is_train=True, aug_strategy=aug_strategy),
    )

    val_ds = ApplyAlbumentations(
        ds=KITTIMOTSToTorchvision(
            datasets.KITTIMOTS(root, split=val_split, seed=seed, split_ratio=split_ratio)
        ),
        tf=get_transforms(is_train=False),
    )

    print(f"Data loaded  |  train: {len(train_ds)}  val: {len(val_ds)}")
    print(f"Mode: {mode}  seed: {seed}  splits: {train_split} / {val_split}")

    num_workers = cfg["data"].get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"].get("val_batch_size", 4),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    classes = ["Background", "Car", "Pedestrian"]

    train_coco_metrics = CocoMetrics(root=root, split=train_split, ann_source="txt", seed=seed, split_ratio=split_ratio)
    val_coco_metrics   = CocoMetrics(root=root, split=val_split,   ann_source="txt", seed=seed, split_ratio=split_ratio)

    return Data(classes, train_loader, val_loader, train_coco_metrics, val_coco_metrics)


def setup_model(exp: Exp, data: Data) -> Run:
    """Instantiate the HuggingFace DETR model, processor, optimizer, and scheduler."""
    cfg        = exp.cfg
    device     = exp.device
    model_name = cfg["model"]["name"]
    weights    = cfg["model"].get("weights", "facebook/detr-resnet-50")

    print(f"Initialising HuggingFace model: {model_name} from {weights}")

    if model_name != "detr":
        raise ValueError(
            f"Unknown model '{model_name}'.  "
            "This script only supports 'detr'.  "
            "Use fine_tune.py for Faster R-CNN or fine_tune_yolo.py for YOLO."
        )

    processor = DetrImageProcessor.from_pretrained(weights)
    
    id2label = {i: label for i, label in enumerate(data.classes)}
    label2id = {label: i for i, label in enumerate(data.classes)}

    model = DetrForObjectDetection.from_pretrained(
        weights,
        num_labels=len(data.classes),
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id,
    )

    # Implement simple Freeze Strategy for DETR
    freeze_strat = cfg["training"].get("freeze_strategy", 1)
    
    if freeze_strat >= 1:
        # Level 1: Freeze backbone
        for param in model.model.backbone.parameters():
            param.requires_grad = False
    if freeze_strat >= 2:
        # Level 2: Freeze transformer encoder
        for param in model.model.encoder.parameters():
            param.requires_grad = False
    if freeze_strat >= 3:
        # Level 3: Freeze transformer decoder (only train prediction heads)
        for param in model.model.decoder.parameters():
            param.requires_grad = False

    model.to(device)

    # Only pass parameters that require gradients to the optimizer.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=cfg["training"]["lr"])
    scheduler = build_scheduler(optimizer, cfg)

    history = {"train_loss": [], "train_map": [], "val_loss": [], "val_map": []}
    return Run(model, processor, optimizer, history, scheduler, best_map=0.0, best_epoch=0)


# ===========================================================================
# Evaluation
# ===========================================================================

# Label mapping from model class IDs to COCO category IDs.
# Model:  1 = Car,    2 = Pedestrian
# COCO:   3 = Car,    1 = Person
_MODEL_TO_COCO_LABEL = {1: 3, 2: 1}


def evaluate(exp: Exp, run: Run, loader: Any, metrics_obj: Any) -> "Eval":
    """Run inference on ``loader`` and compute COCO detection metrics."""
    model     = run.model
    processor = run.processor
    device    = exp.device

    if model is None:
        return Eval(predictions=[], metrics={"coco/AP": 0.0})

    model.eval()
    coco_dt_list: List[Dict] = []

    with torch.no_grad():
        for images, targets in loader:
            # Process inputs for DETR (padding and normalization)
            # do_rescale=False since images are already [0, 1] tensors
            batch_dict = processor(images=images, return_tensors="pt", do_rescale=False)
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            
            outputs = model(**batch_dict)
            
            # Post-process outputs to standard xyxy bounding boxes
            target_sizes = torch.tensor([img.shape[1:] for img in images]).to(device)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)

            for res, tgt in zip(results, targets):
                image_id = int(tgt["image_id"].item())

                for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
                    cat_id = int(label.item())
                    if cat_id not in _MODEL_TO_COCO_LABEL:
                        continue  # ignore unknown classes

                    coco_dt_list.append({
                        "image_id":   image_id,
                        "category_id": _MODEL_TO_COCO_LABEL[cat_id],
                        "bbox":        _xyxy_to_xywh(box.cpu().tolist()),
                        "score":       float(score.item()),
                    })

    if not coco_dt_list:
        print("Warning: no predictions were generated for this split.")
        return Eval(predictions=[], metrics={"coco/AP": 0.0})

    coco_gt      = metrics_obj.coco_gt
    coco_dt      = coco_gt.loadRes(coco_dt_list)
    metrics_result = metrics_obj.compute_metrics(coco_dt)

    return Eval(predictions=coco_dt_list, metrics=metrics_result)


def log_predictions_to_wandb(
    exp: Exp,
    data: Data,
    run: Run,
    epoch: int,
    max_images: int = 4,
) -> None:
    """Upload a sample of validation images with predicted and GT boxes to W&B."""
    model     = run.model
    processor = run.processor
    device    = exp.device

    if model is None:
        return

    model.eval()

    CLASS_NAMES = {1: "Car", 2: "Pedestrian"}

    dataset_len     = len(data.val_loader.dataset)
    step            = max(1, dataset_len // max_images)
    target_indices  = set(range(0, dataset_len, step))
    logged_images: List[wandb.Image] = []
    current_idx = 0

    with torch.no_grad():
        for images, targets in data.val_loader:
            batch_size = len(images)

            log_in_batch = [
                i for i in range(batch_size)
                if current_idx + i in target_indices
            ][:max_images - len(logged_images)]

            if not log_in_batch:
                current_idx += batch_size
                continue

            # Process only the selected subset to save compute
            subset_images = [images[i] for i in log_in_batch]
            batch_dict = processor(images=subset_images, return_tensors="pt", do_rescale=False)
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            
            outputs = model(**batch_dict)
            target_sizes = torch.tensor([img.shape[1:] for img in subset_images]).to(device)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.25)

            for res, i in zip(results, log_in_batch):
                img_np = (images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                tgt    = targets[i]

                # --- Predicted boxes ---
                wandb_pred_boxes = [
                    {
                        "position":    {"minX": float(b[0]), "minY": float(b[1]),
                                        "maxX": float(b[2]), "maxY": float(b[3])},
                        "class_id":    int(l),
                        "box_caption": f"{CLASS_NAMES.get(int(l), 'Unknown')} {s:.2f}",
                        "scores":      {"score": float(s)},
                        "domain":      "pixel",
                    }
                    for b, l, s in zip(res["boxes"], res["labels"], res["scores"])
                ]

                # --- Ground-truth boxes ---
                wandb_gt_boxes = [
                    {
                        "position":    {"minX": float(b[0]), "minY": float(b[1]),
                                        "maxX": float(b[2]), "maxY": float(b[3])},
                        "class_id":    int(l),
                        "box_caption": f"{CLASS_NAMES.get(int(l), 'Unknown')} (GT)",
                        "domain":      "pixel",
                    }
                    for b, l in zip(tgt["boxes"].cpu().numpy(), tgt["labels"].cpu().numpy())
                ]

                logged_images.append(wandb.Image(img_np, boxes={
                    "predictions":  {"box_data": wandb_pred_boxes, "class_labels": CLASS_NAMES},
                    "ground_truth": {"box_data": wandb_gt_boxes,   "class_labels": CLASS_NAMES},
                }))

            current_idx += batch_size
            if len(logged_images) >= max_images:
                break

    if logged_images:
        wandb.log({"val_predictions": logged_images}, commit=False)


# ===========================================================================
# Training loop
# ===========================================================================

def train(exp: Exp, data: Data, run: Run) -> Run:
    """Run the full training loop and persist the best checkpoint."""
    cfg             = exp.cfg
    device          = exp.device
    best_model_path = exp.best_model_path
    model           = run.model
    processor       = run.processor
    optimizer       = run.optimizer
    scheduler       = run.scheduler
    history         = run.history
    best_map        = run.best_map
    best_epoch      = run.best_epoch
    num_epochs      = cfg["training"]["epochs"]

    if model is None:
        print("Model is None — skipping training.")
        return run

    print("Starting training…")

    for epoch in tqdm(range(num_epochs), desc="Epochs", mininterval=60, ascii=True):

        # ---- Training pass ----
        model.train()
        train_loss_sum = 0.0

        for images, targets in data.train_loader:
            # Format targets for DetrImageProcessor
            formatted_targets = [
                {"boxes": t["boxes"], "class_labels": t["labels"]}
                for t in targets
            ]
            
            # do_rescale=False avoids dividing already [0, 1] tensors by 255
            batch_dict = processor(images=images, annotations=formatted_targets, return_tensors="pt", do_rescale=False)
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

            outputs = model(**batch_dict)
            loss    = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())

        train_loss = train_loss_sum / max(1, len(data.train_loader))

        # ---- Validation loss pass ----
        model.eval()  # Keep eval mode for val loss to prevent batchnorm/dropout updates
        val_loss_sum = 0.0

        with torch.no_grad():
            for images, targets in data.val_loader:
                formatted_targets = [
                    {"boxes": t["boxes"], "class_labels": t["labels"]}
                    for t in targets
                ]
                batch_dict = processor(images=images, annotations=formatted_targets, return_tensors="pt", do_rescale=False)
                batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
                
                outputs = model(**batch_dict)
                val_loss_sum += float(outputs.loss.item())

        val_loss = val_loss_sum / max(1, len(data.val_loader))

        # ---- COCO evaluation (switch to eval() mode internally) ----
        eval_result = evaluate(exp, run, data.val_loader, data.val_coco_metrics)
        val_map     = eval_result.metrics["overall/AP"]

        # ---- Append to history ----
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map"].append(val_map)

        # ---- Optional prediction image logging ----
        log_interval = cfg["training"].get("log_images_every", 5)
        if (epoch + 1) % log_interval == 0:
            log_predictions_to_wandb(exp, data, run, epoch=epoch + 1)

        print(
            f"Epoch {epoch + 1}/{num_epochs}  |  "
            f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
            f"val_COCO_AP: {val_map:.4f}"
        )

        # ---- LR scheduler step ----
        if scheduler is not None:
            scheduler.step()

        # ---- Checkpoint best model ----
        if val_map > best_map:
            best_map   = val_map
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New best: COCO AP {best_map:.4f} at epoch {epoch + 1}")

            with open(os.path.join(exp.output_dir, "best_metrics.json"), "w") as fh:
                json.dump(eval_result.metrics, fh, indent=4)

            with open(os.path.join(exp.output_dir, "best_predictions.jsonl"), "w") as fh:
                for pred in eval_result.predictions:
                    fh.write(json.dumps(pred) + "\n")

        # ---- W&B logging ----
        wandb_log = {
            "epoch":       epoch + 1,
            "train_loss":  train_loss,
            "val_loss":    val_loss,
            "val_coco_ap": val_map,
            "best_map":    best_map,
            "best_epoch":  best_epoch,
        }
        wandb_log.update({f"val_{k}": v for k, v in eval_result.metrics.items()})
        wandb.log(wandb_log)

        # ---- CSV logging ----
        csv_path = os.path.join(exp.output_dir, "metrics_history.csv")
        write_header = not os.path.isfile(csv_path)
        headers = ["epoch", "train_loss", "val_loss", "best_map"] + [
            f"val_{k}" for k in eval_result.metrics.keys()
        ]

        with open(csv_path, "a") as fh:
            if write_header:
                fh.write(",".join(headers) + "\n")
            row = (
                [str(epoch + 1), f"{train_loss:.6f}", f"{val_loss:.6f}", f"{best_map:.6f}"]
                + [f"{v:.6f}" for v in eval_result.metrics.values()]
            )
            fh.write(",".join(row) + "\n")

    print(f"Training complete.  Best COCO AP: {best_map:.4f} at epoch {best_epoch + 1}.")
    run.best_map   = best_map
    run.best_epoch = best_epoch
    run.history    = history
    return run


# ===========================================================================
# Entry point
# ===========================================================================

def main(args: argparse.Namespace) -> None:
    """Orchestrate the full fine-tuning pipeline."""
    exp  = setup_experiment(args.config, args)
    data = setup_data(exp)
    run  = setup_model(exp, data)
    run  = train(exp, data, run)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DETR object detection model on KITTI-MOTS.")

    # Required
    parser.add_argument("--config",  type=str, required=True, help="Path to YAML config file.")

    # Hyper-parameter overrides (used by W&B sweeps)
    parser.add_argument("--epochs",     type=int,   help="Override training.epochs.")
    parser.add_argument("--batch_size", type=int,   help="Override training.batch_size.")
    parser.add_argument("--lr",         type=float, help="Override training.lr.")
    parser.add_argument("--t_max",      type=int,   help="Override training.scheduler.t_max.")

    # Experiment naming / tracking
    parser.add_argument("--project", type=str, help="Override W&B project name.")
    parser.add_argument("--name",    type=str, help="Override W&B run / experiment name.")

    # Data split control
    parser.add_argument(
        "--mode", type=str, choices=["full", "search"],
        help="'full': train on all training data, eval on official val. "
             "'search': use a seeded sub-split for hyper-parameter search.",
    )
    parser.add_argument("--seed",        type=int,   help="Random seed for data splitting.")
    parser.add_argument("--split_ratio", type=float, help="Fraction of training data kept in 'search' mode.")

    # Model / training strategy overrides
    parser.add_argument(
        "--freeze_strategy", type=int,
        help="Backbone freeze level (1–3). Level 1: Backbone, 2: Backbone+Encoder, 3: Backbone+Encoder+Decoder.",
    )
    parser.add_argument(
        "--aug_strategy", type=str,
        help="Augmentation strategy name (base, legacy, geometric, color_jitter, "
             "extreme_weather, heavy_corruption, limit_test).",
    )
    parser.add_argument(
        "--name_fields", type=str,
        help="Comma-separated config fields to include in the auto-generated run name "
             "(e.g. 'freeze_strategy,lr').",
    )

    args = parser.parse_args()
    main(args)