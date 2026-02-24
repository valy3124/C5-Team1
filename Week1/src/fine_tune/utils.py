"""
utils.py — Shared utilities for fine-tuning DETR and Faster R-CNN.
Contains standard dataset adapters, augmentations, and experiment setup logic.
"""

import os
import sys
import argparse
import yaml
import torch
import wandb
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
def bootstrap_paths():
    """Add src/ and Week1/ to sys.path so we can import project modules."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir     = os.path.dirname(current_dir)
    week1_dir   = os.path.dirname(src_dir)
    for p in (src_dir, week1_dir):
        if p not in sys.path:
            sys.path.append(p)

# Call immediately so other modules can import `datasets`
bootstrap_paths()
import datasets
from inference.evaluation import CocoMetrics

# ===========================================================================
# Data Containers
# ===========================================================================
@dataclass
class Exp:
    cfg:             Dict[str, Any]
    device:          Any
    output_dir:      str
    best_model_path: str

@dataclass
class Data:
    classes:            List[str]
    train_loader:       Any
    val_loader:         Any
    train_coco_metrics: Any = None
    val_coco_metrics:   Any = None

@dataclass
class Run:
    model:      Any
    optimizer:  Any
    history:    Dict[str, list]
    processor:  Any = None        # Used by DETR, None for Faster R-CNN
    scheduler:  Any = None
    best_map:   float = 0.0
    best_epoch: int   = 0

@dataclass
class Eval:
    predictions: List[Dict]
    metrics:     Dict[str, float]
    coco_eval:   Any = None
    inference_fps: float = 0.0          
    inference_latency_ms: float = 0.0 

# ===========================================================================
# Utility Helpers
# ===========================================================================
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def _xyxy_to_xywh(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

def get_common_parser(description: str) -> argparse.ArgumentParser:
    """Returns a pre-populated ArgumentParser with the standard arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config",  type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--epochs",     type=int,   help="Override training.epochs.")
    parser.add_argument("--batch_size", type=int,   help="Override training.batch_size.")
    parser.add_argument("--lr",         type=float, help="Override training.lr.")
    parser.add_argument("--t_max",      type=int,   help="Override training.scheduler.t_max.")
    parser.add_argument("--project", type=str, help="Override W&B project name.")
    parser.add_argument("--name",    type=str, help="Override W&B run / experiment name.")
    parser.add_argument("--mode", type=str, choices=["full", "search"], help="'full' or 'search'")
    parser.add_argument("--seed",        type=int,   help="Random seed for data splitting.")
    parser.add_argument("--split_ratio", type=float, help="Fraction of training data kept in 'search' mode.")
    parser.add_argument("--freeze_strategy", type=int, help="Backbone freeze level.")
    parser.add_argument("--aug_strategy", type=str, help="Augmentation strategy name.")
    parser.add_argument("--name_fields", type=str, help="Fields for auto-generated run name.")
    parser.add_argument("--nms_iou_threshold", type=float, help="Override NMS IoU threshold (Faster R-CNN).")
    parser.add_argument("--optimizer", type=str, help="Override optimizer.")
    parser.add_argument("--scheduler", type=str, help="Override scheduler name.")
    parser.add_argument("--gradient_clipping", type=float, help="Override gradient clipping max norm.")
    return parser

# ===========================================================================
# Setup Functions
# ===========================================================================
def setup_experiment(config_path: str, args: argparse.Namespace) -> Exp:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    training = cfg["training"]
    if args.epochs:     training["epochs"]     = args.epochs
    if args.batch_size: training["batch_size"] = args.batch_size
    if args.lr:         training["lr"]         = args.lr
    if args.t_max:      training.setdefault("scheduler", {})["t_max"] = args.t_max
    if args.project: cfg["project"]         = args.project
    if args.name:    cfg["experiment_name"] = args.name
    
    if getattr(args, "freeze_strategy", None) is not None: training["freeze_strategy"]  = int(args.freeze_strategy)
    if getattr(args, "aug_strategy", None) is not None: training["aug_strategy"]     = str(args.aug_strategy)
    if getattr(args, "nms_iou_threshold", None) is not None: training["nms_iou_threshold"] = float(args.nms_iou_threshold)
    if getattr(args, "optimizer", None) is not None: training["optimizer"] = str(args.optimizer)
    if getattr(args, "scheduler", None) is not None: training.setdefault("scheduler", {})["name"] = str(args.scheduler)
    if getattr(args, "gradient_clipping", None) is not None: training["gradient_clipping"] = float(args.gradient_clipping)

    cfg.setdefault("data", {})
    cfg["data"]["mode"]        = args.mode        or cfg["data"].get("mode", "full")
    cfg["data"]["seed"]        = args.seed        or cfg["data"].get("seed", 42)
    cfg["data"]["split_ratio"] = args.split_ratio or cfg["data"].get("split_ratio", 0.8)

    if not args.name:
        parts = [cfg["model"].get("name", "model")]
        
        aug = training.get("aug_strategy", "base")
        parts.append(str(aug))
        
        if "freeze_strategy" in training:
            parts.append(f"Frz{training['freeze_strategy']}")
            
        if "lr" in training:
            parts.append(f"LR{training['lr']:.1e}")
            
        if "weight_decay" in training:
            parts.append(f"WD{training['weight_decay']:.1e}")
            
        if "optimizer" in training:
            parts.append(f"Opt_{training['optimizer']}")
            
        if "scheduler" in training and training["scheduler"].get("name", "none") != "none":
            parts.append(f"Sch_{training['scheduler']['name']}")
            
        if "gradient_clipping" in training:
            parts.append(f"GC_{training['gradient_clipping']:.1f}")

        cfg["experiment_name"] = "_".join(parts)
        print(f"Auto-generated experiment name: {cfg['experiment_name']}")

    wandb.init(project=cfg.get("project", "kitti-mots-finetune"), name=cfg.get("experiment_name", "run"), config=cfg)

    model_dir  = cfg["model"].get("name", "unknown_model")
    output_dir = os.path.join(cfg.get("output_dir", "results"), model_dir, f"{cfg.get('experiment_name', 'exp')}_{wandb.run.id}")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.yaml"), "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return Exp(cfg, device, output_dir, os.path.join(output_dir, "best_model.pth"))

def build_scheduler(optimizer, cfg: Dict[str, Any]) -> Optional[Any]:
    sch_cfg = cfg["training"].get("scheduler")
    if sch_cfg is None: return None
    name = sch_cfg.get("name", "none").lower()
    if name in ("none", "null"): return None
    if name in ("step", "steplr"):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_cfg.get("step_size", 3), gamma=sch_cfg.get("gamma", 0.1))
    if name in ("cosine", "cosineannealing", "cosineannealinglr"):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sch_cfg.get("t_max", cfg["training"].get("epochs", 10)), eta_min=sch_cfg.get("eta_min", 0.0))
    raise ValueError(f"Unknown scheduler '{name}'.")


# ===========================================================================
# Dataset Adapters
# ===========================================================================
class KITTIMOTSToTorchvision(torch.utils.data.Dataset):
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
        boxes, labels = [], []
        for ann in raw_anns:
            cls = int(getattr(ann, "class_id", -1))
            box = getattr(ann, "bbox_xyxy", None)
            if box is None: continue
            x1, y1, x2, y2 = map(float, box)
            x2 += 1
            y2 += 1
            if x2 <= x1 or y2 <= y1: continue
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        area = ((boxes_t[:, 2] - boxes_t[:, 0]).clamp(min=0) * (boxes_t[:, 3] - boxes_t[:, 1]).clamp(min=0)) if boxes_t.numel() else torch.zeros((0,), dtype=torch.float32)
        return {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area":     area.to(torch.float32),
            "iscrowd":  torch.zeros((labels_t.shape[0],), dtype=torch.int64),
        }

class ApplyAlbumentations(torch.utils.data.Dataset):
    def __init__(self, ds: torch.utils.data.Dataset, tf: Optional[A.Compose], keep_empty: bool = True) -> None:
        self.ds = ds
        self.tf = tf
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
            if isinstance(img, torch.Tensor):
                img = img.float().div(255.0) if img.dtype == torch.uint8 else img.float()
            else:
                img = to_tensor(img)
        else:
            img       = to_tensor(img_np)
            boxes_tf  = target["boxes"].numpy().tolist()
            labels_tf = target["labels"].numpy().tolist()

        if len(boxes_tf) == 0 and not self.keep_empty:
            return self.__getitem__((idx + 1) % len(self))

        if boxes_tf:
            target["boxes"]  = torch.tensor(boxes_tf,  dtype=torch.float32)
            target["labels"] = torch.tensor(labels_tf, dtype=torch.int64)
        else:
            target["boxes"]  = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,),   dtype=torch.int64)

        target["area"] = ((target["boxes"][:, 2] - target["boxes"][:, 0]).clamp(min=0) * (target["boxes"][:, 3] - target["boxes"][:, 1]).clamp(min=0)) if target["boxes"].numel() else torch.zeros((0,), dtype=torch.float32)
        return img, target

# ===========================================================================
# Augmentation Strategies
# ===========================================================================
def get_transforms(is_train: bool = True, aug_strategy: str = "base") -> A.Compose:
    bbox_params = A.BboxParams(format="pascal_voc", label_fields=["class_labels"], clip=True, min_area=1, min_visibility=0.1)
    if not is_train:
        return A.Compose([ToTensorV2()], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], clip=True))

    if aug_strategy == "legacy":
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
        tfms = [A.HorizontalFlip(p=0.5)]
        if aug_strategy in ("geometric", "limit_test"):
            tfms += [A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15, border_mode=0, p=0.5), A.Perspective(scale=(0.05, 0.1), p=0.3)]
        if aug_strategy in ("color_jitter", "limit_test"):
            tfms += [A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5), A.HueSaturationValue(p=0.3), A.RandomGamma(gamma_limit=(80, 120), p=0.3), A.RGBShift(p=0.2)]
        if aug_strategy in ("extreme_weather", "limit_test"):
            prob = 0.5 if aug_strategy == "limit_test" else 0.8
            tfms += [A.OneOf([A.RandomRain(brightness_coefficient=0.7, drop_width=1, blur_value=5, p=1), A.RandomFog(fog_coef_range=(0.4, 0.9), alpha_coef=0.1, p=1), A.RandomShadow(shadow_limit=(1, 3), p=1), A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_limit=(0, 1), p=1)], p=prob)]
        if aug_strategy in ("heavy_corruption", "limit_test"):
            tfms += [A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), A.MotionBlur(blur_limit=(3, 7), p=0.3), A.CoarseDropout(num_holes_range=(2, 10), hole_height_range=(10, 40), hole_width_range=(10, 40), fill=0, p=0.4), A.ImageCompression(quality_range=(50, 90), p=0.3)]
    
    tfms.append(ToTensorV2())
    return A.Compose(tfms, bbox_params=bbox_params)

def setup_data(exp: Exp) -> Data:
    cfg = exp.cfg
    root = cfg["data"]["root"]
    mode = cfg["data"].get("mode", "full")
    seed = cfg["data"].get("seed", 42)
    split_ratio = cfg["data"].get("split_ratio", 0.8)
    aug_strategy = cfg["training"].get("aug_strategy", "legacy")

    if mode == "full": train_split, val_split = "train_full", "validation"
    elif mode == "search": train_split, val_split = "train", "dev"
    else: raise ValueError(f"Unknown data.mode '{mode}'.")

    train_ds = ApplyAlbumentations(
        ds=KITTIMOTSToTorchvision(datasets.KITTIMOTS(root, split=train_split, ann_source="txt", seed=seed, split_ratio=split_ratio)),
        tf=get_transforms(is_train=True, aug_strategy=aug_strategy),
    )
    val_ds = ApplyAlbumentations(
        ds=KITTIMOTSToTorchvision(datasets.KITTIMOTS(root, split=val_split, seed=seed, split_ratio=split_ratio)),
        tf=get_transforms(is_train=False),
    )

    num_workers = cfg["data"].get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"].get("val_batch_size", 4), shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    classes = ["Background", "Car", "Pedestrian"]
    train_coco_metrics = CocoMetrics(root=root, split=train_split, ann_source="txt", seed=seed, split_ratio=split_ratio)
    val_coco_metrics   = CocoMetrics(root=root, split=val_split,   ann_source="txt", seed=seed, split_ratio=split_ratio)

    return Data(classes, train_loader, val_loader, train_coco_metrics, val_coco_metrics)