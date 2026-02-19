import argparse
import os
import sys

# --- 1. Setup the Path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. Standard Imports ---
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import to_tensor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import wandb
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# --- 3. Project Imports ---
import datasets
from models.faster_rcnn import FasterRCNNModel


# TODO: Benet, si podem juntar el dos adapters de sota per a fer-ho més modulable amb
# tot el codi seria perf. Potser posar a algun altre lloc
# KITTI-MOTS -> Torchvision adapter
class KITTIMOTSToTorchvision(torch.utils.data.Dataset):
    """
    Returns:
      image: FloatTensor[C,H,W] in [0,1]
      target: dict(boxes, labels, image_id, area, iscrowd)
    """
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def raw_anns_to_target(self, raw_anns: List[Any], image_id: int) -> Dict[str, torch.Tensor]:
        boxes, labels = [], []

        for ann in raw_anns:
            cls = int(getattr(ann, "class_id", -1))
            box = getattr(ann, "bbox_xyxy", None)
            if box is None:
                continue

            boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
            labels.append(int(cls))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)

        area = (
            (boxes_t[:, 2] - boxes_t[:, 0]).clamp(min=0) *
            (boxes_t[:, 3] - boxes_t[:, 1]).clamp(min=0)
        ) if boxes_t.numel() else torch.zeros((0,), dtype=torch.float32)

        return {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area.to(torch.float32),
            "iscrowd": torch.zeros((labels_t.shape[0],), dtype=torch.int64)}

    def __getitem__(self, idx: int):
        img_pil, raw_anns, _ = self.base_ds[idx]
        img_np = np.array(img_pil)
        target = self.raw_anns_to_target(raw_anns, image_id=idx)
        return img_np, target


# Apply Transforms to each image
class ApplyAlbumentations(torch.utils.data.Dataset):
    def __init__(self, ds, tf, keep_empty: bool = True):
        self.ds = ds # already adapted
        self.tf = tf
        self.keep_empty = keep_empty
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img_np, target = self.ds[idx]

        boxes = target["boxes"].numpy().tolist()
        labels = target["labels"].numpy().tolist()

        if self.tf is not None:
            out = self.tf(image=img_np, bboxes=boxes, class_labels=labels)
            img = out["image"]
            boxes_tf = out["bboxes"]
            labels_tf = out["class_labels"]

            if isinstance(img, torch.Tensor):
                if img.dtype == torch.uint8:
                    img = img.float().div(255.0)
                else:
                    img = img.float()
            else:
                # safety fallback if transform didn't convert to tensor
                img = to_tensor(img)  # -> float [0,1]
        else:
            img = to_tensor(img_np)
            boxes_tf, labels_tf = boxes, labels

        # after albumentations the boxes could be removed...avoid them
        if len(boxes_tf) == 0 and not self.keep_empty:
            return self.__getitem__((idx + 1) % len(self))

        if len(boxes_tf) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        else:
            target["boxes"] = torch.tensor(boxes_tf, dtype=torch.float32)
            target["labels"] = torch.tensor(labels_tf, dtype=torch.int64)

        target["area"] = (
            (target["boxes"][:, 2] - target["boxes"][:, 0]).clamp(min=0) *
            (target["boxes"][:, 3] - target["boxes"][:, 1]).clamp(min=0)
        ) if target["boxes"].numel() else torch.zeros((0,), dtype=torch.float32)

        return img, target


@dataclass
class Exp:
    """Container for experiment-level configuration and environment state."""

    cfg: Dict[str, Any]          # Parsed experiment configuration dictionary
    device: Any                  # Torch device used for training and evaluation
    output_dir: str              # Directory where results and artifacts are stored
    best_model_path: str         # Path for saving the best model weights


@dataclass
class Data:
    """Container for dataset-related objects and loaders."""

    classes: List[str]           # List of class names
    train_loader: Any            # DataLoader for the training set
    val_loader: Any              # DataLoader for the val set


@dataclass
class Run:
    """Container for model, optimization, and training state."""

    model: Any                   # Neural network model instance
    optimizer: Any               # Optimizer instance
    history: Dict[str, list]     # Dictionary storing training and test metrics
    scheduler: Any = None        # Optional scheduler
    best_map: float = 0.0        # Best mAP achieved so far
    best_epoch: int = 0          # Epoch corresponding to best mAP


@dataclass
class Eval:
    """Container for evaluation outputs."""
    predictions: List[Dict]
    metrics: Dict[str, float]


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def setup_experiment(config_path: str) -> Exp:
    """Initialize experiment configuration, logging, and environment.

    Loads the YAML configuration file, initializes Weights & Biases logging,
    sets the output directory, and selects the compute device.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Exp object containing configuration and environment information.
    """

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # WandB init
    wandb.init(project=cfg.get('project', 'kitti-mots-finetune'), 
               name=cfg.get('experiment_name', 'run'),
               config=cfg)

    output_dir = os.path.join(cfg.get('output_dir', 'results'), f"{cfg.get('experiment_name', 'exp')}_{wandb.run.id}")
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    best_model_path = os.path.join(output_dir, "best_model.pth")
    return Exp(cfg, device, output_dir, best_model_path)


def setup_data(exp: Exp) -> Data:
    """Initialize datasets and data loaders (Train and Test only).

    Args:
        exp: Experiment configuration and environment information.

    Returns:
        Data object containing datasets and data loaders.
    """

    cfg = exp.cfg
    root = cfg['data']['root'] 
    
    # Data loading
    train_base = datasets.KITTIMOTS(root, split="train", ann_source="txt")
    train_ds = ApplyAlbumentations(KITTIMOTSToTorchvision(train_base), tf=get_transforms(True))

    val_base = datasets.KITTIMOTS(root, split="validation")
    val_ds = ApplyAlbumentations(KITTIMOTSToTorchvision(val_base), tf=get_transforms(False))

    print(f"Data Loaded: {len(train_ds)} Train, {len(val_ds)} Val")

    train_loader = DataLoader(train_ds, 
                              batch_size=cfg['training']['batch_size'], 
                              shuffle=True, collate_fn=collate_fn, 
                              num_workers=cfg['data'].get('num_workers', 4))
    
    val_loader = DataLoader(val_ds, 
                            batch_size=1, 
                            shuffle=False, collate_fn=collate_fn, 
                            num_workers=cfg['data'].get('num_workers', 4))
    
    classes = ["Background", "Car", "Pedestrian"]
    return Data(classes, train_loader, val_loader)


def build_scheduler(optimizer, cfg):
    sch_cfg = cfg["training"].get("scheduler")
    if sch_cfg is None:
        return None
    
    name = sch_cfg.get("name", "none").lower()

    if name in ["none", "null"]:
        return None

    if name in ["step", "steplr"]:
        step_size = sch_cfg.get("step_size", 3)
        gamma = sch_cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name in ["cosine", "cosineannealing", "cosineannealinglr"]:
        T_max = sch_cfg.get("t_max", cfg["training"].get("epochs", 10))
        eta_min = sch_cfg.get("eta_min", 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    raise ValueError(f"Unknown scheduler: {name}")


def setup_model(exp: Exp, data: Data) -> Run:
    """Initialize model and optimizer.

    Args:
        exp: Experiment configuration and environment information.
        data: Data object containing datasets and data loaders.

    Returns:
        Run object containing model and optimizer.
    """
    cfg = exp.cfg
    device = exp.device
    model_name = cfg['model']['name']
    
    print(f"Initializing Model: {model_name}")
    
    if model_name == "faster_rcnn":        
        # Initialize the wrapper (loads backbone/weights)
        model = FasterRCNNModel(device=str(device))
        num_classes = len(data.classes)
        train_backbone = cfg['training'].get('train_backbone', False)
        model.prepare_finetune(num_classes, train_backbone)
        
        # Optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=cfg['training']['lr'])
        
        # Scheduler
        scheduler = build_scheduler(optimizer, cfg)

    elif model_name == "detr":
        # TODO: Wait for DeTR implementation
        print("DeTR not fully implemented yet.")
        model = None; optimizer = None; scheduler = None

    elif model_name == "yolo":
        # TODO: Wait for YOLO implementation
        print("YOLO not fully implemented yet.")
        model = None; optimizer = None; scheduler = None

    else:
        raise ValueError(f"Unknown model: {model_name}")

    history = {'train_loss': [], 'train_map': [], 'val_loss': [], 'val_map': []}
    
    return Run(model, optimizer, history, scheduler, best_map=0.0, best_epoch=0)

# Albumentations transforms for Faster-RCNN
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.02, scale_limit=0.10, rotate_limit=5,
                border_mode=0, p=0.5
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.2),
            A.MotionBlur(p=0.1),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'], 
            clip=True, 
            min_area=1,
            min_visibility=0.1
            )
        )
    else:
        return A.Compose([
            ToTensorV2()
        ], 
        bbox_params=A.BboxParams(
            format='pascal_voc', 
            label_fields=['class_labels'],
            clip=True
            )
        )


def _to_xywh(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def evaluate(exp: Exp, data: Data, run: Run) -> Eval:
    cfg = exp.cfg
    device = exp.device
    model = run.model
    val_loader = data.val_loader

    if model is None:
        return Eval([], {"coco/AP": 0.0})

    # Load best weights if available
    if os.path.exists(exp.best_model_path):
        model.load_state_dict(torch.load(exp.best_model_path, map_location=device))

    model.eval()

    # Build COCO GT dict
    coco_gt_dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "car"},
            {"id": 2, "name": "pedestrian"},
        ],
    }

    coco_dt_list = []
    ann_id = 1

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="COCO eval"):
            img = images[0].to(device)
            tgt = targets[0]

            image_id = int(tgt["image_id"].item())
            H, W = img.shape[-2], img.shape[-1]

            coco_gt_dict["images"].append({
                "id": image_id,
                "width": int(W),
                "height": int(H),
                "file_name": str(image_id),
            })

            # GT annotations
            gt_boxes = tgt["boxes"].cpu().numpy()
            gt_labels = tgt["labels"].cpu().numpy()

            for b, c in zip(gt_boxes, gt_labels):
                # IMPORTANT: COCO category_id must match your label IDs (1,2)
                cat_id = int(c)
                if cat_id == 0:
                    continue  # background should never appear, but just in case

                x, y, w, h = _to_xywh(b)
                area = float(max(w, 0.0) * max(h, 0.0))

                coco_gt_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0,
                })
                ann_id += 1

            # Predictions (no targets in eval)
            preds = model([img])[0]
            pb = preds["boxes"].detach().cpu().numpy()
            pl = preds["labels"].detach().cpu().numpy()
            ps = preds["scores"].detach().cpu().numpy()

            for b, c, s in zip(pb, pl, ps):
                cat_id = int(c)
                x, y, w, h = _to_xywh(b)
                coco_dt_list.append({
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x, y, w, h],
                    "score": float(s),
                })

    # Run COCOeval
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(coco_dt_list) if len(coco_dt_list) else coco_gt.loadRes([])

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats  # 12 numbers
    metrics = {
        "coco/AP": float(stats[0]),         # AP @[.5:.95]
        "coco/AP50": float(stats[1]),       # AP @0.5
        "coco/AP75": float(stats[2]),       # AP @0.75
        "coco/AP_small": float(stats[3]),
        "coco/AP_medium": float(stats[4]),
        "coco/AP_large": float(stats[5]),
        "coco/AR1": float(stats[6]),
        "coco/AR10": float(stats[7]),
        "coco/AR100": float(stats[8]),
        "coco/AR_small": float(stats[9]),
        "coco/AR_medium": float(stats[10]),
        "coco/AR_large": float(stats[11]),
    }

    wandb.log(metrics)
    return Eval(predictions=coco_dt_list, metrics=metrics)


def train(exp: Exp, data: Data, run: Run) -> Run:
    """Train the model and save best model based on mAP.

    Args:
        exp: Experiment configuration and environment information.
        data: Data object containing datasets and data loaders.
        run: Run object containing model and optimizer.

    Returns:
        Run object containing model and optimizer.
    """
    # TODO: check it does not crash with defined models
    cfg = exp.cfg
    device = exp.device
    best_model_path = exp.best_model_path

    train_loader = data.train_loader
    val_loader = data.val_loader

    model = run.model
    optimizer = run.optimizer
    history = run.history
    best_map = run.best_map
    best_epoch = run.best_epoch
    
    if run.model is None:
        print("Model is None, returning.")
        return run

    print("Starting training...")
    for epoch in tqdm(range(cfg['training']['epochs']), desc="TRAINING THE MODEL"):
        model.train()
        train_loss_sum = 0.0

        prog_bar = tqdm(data.train_loader, desc=f"Epoch {epoch+1}")
        for images, targets in prog_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss_sum += float(losses.item())
            prog_bar.set_postfix(loss=float(losses.item()))
        
        train_loss = train_loss_sum / max(1, len(train_loader))

        # torchvision detectors return losses only in train() mode
        model.train()
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} [val-loss]"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
                val_loss_sum += float(losses.item())
        val_loss = val_loss_sum / max(1, len(val_loader))

        # ---- VAL COCO METRICS ----
        eval_ = evaluate(exp, data, run)   # uses val_loader internally
        val_map = eval_.metrics["coco/AP"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map"].append(val_map)

        print(
            f"Epoch {epoch+1}/{cfg['training']['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val COCO AP: {val_map:.4f}"
        )

        if run.scheduler is not None:
            run.scheduler.step()

        if val_map > best_map:
            best_map = val_map
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model: COCO AP {best_map:.4f} at Epoch {epoch+1}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_coco_ap": val_map,
            "best_map": best_map,
            "best_epoch": best_epoch,
        })

    print(f"Training finished. Best model: COCO AP {best_map:.4f} at Epoch {best_epoch+1}")
    run.best_map = best_map
    run.best_epoch = best_epoch
    run.history = history
    return run

def main(config_path: str) -> None:
    exp = setup_experiment(config_path)
    data = setup_data(exp)
    run = setup_model(exp, data)
    run = train(exp, data, run)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
