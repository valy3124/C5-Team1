import argparse
import os
import sys
import json

# --- 1. Setup the Path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
week1_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)
if week1_dir not in sys.path:
    sys.path.append(week1_dir)

# --- 2. Standard Imports ---
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import to_tensor
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
from inference.evaluation import CocoMetrics



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

            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            
            # KITTI MOTS dataset returns inclusive indices for min/max. 
            # We convert to x2,y2 exclusive for compatibility with width calculation (x2-x1) > 0
            x2 += 1
            y2 += 1
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            boxes.append([x1, y1, x2, y2])
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


# Albumentations transforms for Faster-RCNN
def get_transforms(is_train=True, augment=True):
    if is_train and augment:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.10, rotate_limit=10,
                border_mode=0, p=0.5
            ),
            # Weather effects
            A.OneOf([
                A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1),
                A.RandomFog(fog_coef_range=(0.3, 1), alpha_coef=0.08, p=1),
                A.RandomShadow(p=1),
            ], p=0.3),
            # Lighting and Color
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(p=0.3),
            # Occlusion
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, p=0.3),
            # Blur
            A.MotionBlur(p=0.2),
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
    coco_metrics: Any = None     # CocoMetrics instance for evaluation


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
    coco_eval: Any = None


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def setup_experiment(config_path: str, args: argparse.Namespace = None) -> Exp:
    """Initialize experiment configuration, logging, and environment.

    Loads the YAML configuration file, initializes Weights & Biases logging,
    sets the output directory, and selects the compute device.

    Args:
        config_path: Path to YAML configuration file.
        args: Parsed command line arguments to override config.

    Returns:
        Exp object containing configuration and environment information.
    """

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override config with args if provided (for sweeps)
    if args:
        if args.epochs:
            cfg['training']['epochs'] = args.epochs
        if args.batch_size:
            cfg['training']['batch_size'] = args.batch_size
        if args.lr:
            cfg['training']['lr'] = args.lr
        
        # Override project/name if provided
        if args.project:
            cfg['project'] = args.project
        if args.name:
            cfg['experiment_name'] = args.name
        
        # New data splitting arguments
        cfg['data'] = cfg.get('data', {})
        if args.mode:
            cfg['data']['mode'] = args.mode
        else:
            cfg['data'].setdefault('mode', 'full') # Default to full if not specified
            
        if args.seed:
            cfg['data']['seed'] = args.seed
        else:
            cfg['data'].setdefault('seed', 42)
            
        if args.split_ratio:
            cfg['data']['split_ratio'] = args.split_ratio
        else:
            cfg['data'].setdefault('split_ratio', 0.8)

        # Sweep overrides for boolean flags (using int 0/1 or str 'true'/'false' for safer sweep passing)
        if args.train_backbone is not None:
            # Handle string/int inputs common in sweeps
            val = str(args.train_backbone).lower()
            cfg['training']['train_backbone'] = (val == 'true' or val == '1')
            
        if args.augment is not None:
             val = str(args.augment).lower()
             cfg['training']['augment'] = (val == 'true' or val == '1')


    # Auto-generate experiment name if not provided (or if it's the default "faster_rcnn_baseline")
    # We check if args.name was provided to override, otherwise generate
    if not args.name:
        aug_str = "Aug" if cfg['training'].get('augment', True) else "NoAug"
        backbone_str = "Backbone" if cfg['training'].get('train_backbone', False) else "Frozen"
        base_name = cfg['model']['name']
        
        # Only auto-name if user didn't explicitly set a custom name in CLI
        # (We assume config file name is generic if we are sweeping)
        cfg['experiment_name'] = f"{base_name}_{aug_str}_{backbone_str}"
        print(f"Auto-generated Experiment Name: {cfg['experiment_name']}")

    # WandB init
    wandb.init(project=cfg.get('project', 'kitti-mots-finetune'), 
               name=cfg.get('experiment_name', 'run'),
               config=cfg)

    output_dir = os.path.join(cfg.get('output_dir', 'results'), f"{cfg.get('experiment_name', 'exp')}_{wandb.run.id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the effective configuration
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
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
    mode = cfg['data'].get('mode', 'full')
    seed = cfg['data'].get('seed', 42)
    split_ratio = cfg['data'].get('split_ratio', 0.8)
    augment = cfg['training'].get('augment', True)

    if mode == "search":
        train_split = "train"
        val_split = "dev"
    elif mode == "full":
        train_split = "train_full"
        val_split = "validation"
    else:
        raise ValueError(f"Invalid mode: {mode}")

    train_base = datasets.KITTIMOTS(root, split=train_split, ann_source="txt",
                                    seed=seed,
                                    split_ratio=split_ratio)
    
    train_ds = ApplyAlbumentations(KITTIMOTSToTorchvision(train_base), tf=get_transforms(is_train=True, augment=augment))

    val_base = datasets.KITTIMOTS(root, split=val_split,
                                  seed=seed,
                                  split_ratio=split_ratio)
    val_ds = ApplyAlbumentations(KITTIMOTSToTorchvision(val_base), tf=get_transforms(is_train=False))

    print(f"Data Loaded: {len(train_ds)} Train, {len(val_ds)} Val")
    print(f"Mode: {mode}, Seed: {seed}, Split: {train_split}/{val_split}")

    train_loader = DataLoader(train_ds, 
                              batch_size=cfg['training']['batch_size'], 
                              shuffle=True, collate_fn=collate_fn, 
                              num_workers=cfg['data'].get('num_workers', 4),
                              pin_memory=True, persistent_workers=True)
    
    val_loader = DataLoader(val_ds, 
                            batch_size=cfg['training'].get('val_batch_size', 4), 
                            shuffle=False, collate_fn=collate_fn, 
                            num_workers=cfg['data'].get('num_workers', 4),
                            pin_memory=True, persistent_workers=True)
    
    classes = ["Background", "Car", "Pedestrian"]
    
    coco_metrics = CocoMetrics(root=root, split=val_split, ann_source="txt", seed=seed, split_ratio=split_ratio)
    
    return Data(classes, train_loader, val_loader, coco_metrics)


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

    model.eval()

    # Build COCO DT list
    coco_dt_list = []
    
    # Mapping: Model class ID -> COCO category ID
    # Model: 1=Car, 2=Pedestrian
    # COCO: 3=Car, 1=Person
    map_label = {1: 3, 2: 1}

    with torch.no_grad():
        for images, targets in val_loader:
            # Batch inference
            imgs = [img.to(device) for img in images]
            preds = model(imgs)

            for i, pred in enumerate(preds):
                tgt = targets[i]
                image_id = int(tgt["image_id"].item())

                pb = pred["boxes"].detach().cpu().numpy()
                pl = pred["labels"].detach().cpu().numpy()
                ps = pred["scores"].detach().cpu().numpy()

                for b, c, s in zip(pb, pl, ps):
                    cat_id = int(c)
                    if cat_id not in map_label:
                        continue 
                    
                    coco_cat_id = map_label[cat_id]
                    x, y, w, h = _to_xywh(b)
                    
                    coco_dt_list.append({
                        "image_id": image_id,
                        "category_id": coco_cat_id,
                        "bbox": [x, y, w, h],
                        "score": float(s),
                    })

    # Run COCOeval using shared logic
    if not coco_dt_list:
        print("No predictions generated.")
        return Eval(predictions=[], metrics={"coco/AP": 0.0})

    coco_gt = data.coco_metrics.coco_gt
    coco_dt = coco_gt.loadRes(coco_dt_list)
    
    metrics = data.coco_metrics.compute_metrics(coco_dt)

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
    for epoch in tqdm(range(cfg['training']['epochs']), desc="TRAINING THE MODEL", mininterval=60, ascii=True):
        model.train()
        train_loss_sum = 0.0

        # prog_bar = tqdm(data.train_loader, desc=f"Epoch {epoch+1}", mininterval=60, ascii=True)
        # for images, targets in prog_bar:
        for images, targets in data.train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss_sum += float(losses.item())
            # prog_bar.set_postfix(loss=float(losses.item()))
        
        train_loss = train_loss_sum / max(1, len(train_loader))

        # torchvision detectors return losses only in train() mode
        model.train()
        val_loss_sum = 0.0
        with torch.no_grad():
            # for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} [val-loss]", mininterval=60, ascii=True):
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
                val_loss_sum += float(losses.item())
        val_loss = val_loss_sum / max(1, len(val_loader))

        # ---- VAL COCO METRICS ----
        eval_ = evaluate(exp, data, run)   # uses val_loader internally
        val_map = eval_.metrics["overall/AP"]

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
            
            # Save metrics
            with open(os.path.join(exp.output_dir, "best_metrics.json"), "w") as f:
                json.dump(eval_.metrics, f, indent=4)
                
            # Save predictions as JSONL
            with open(os.path.join(exp.output_dir, "best_predictions.jsonl"), "w") as f:
                for p in eval_.predictions:
                    f.write(json.dumps(p) + "\n")

        wandb_log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_coco_ap": val_map,
            "best_map": best_map,
            "best_epoch": best_epoch,
        }
        wandb_log_dict.update(eval_.metrics)
        wandb.log(wandb_log_dict)

    print(f"Training finished. Best model: COCO AP {best_map:.4f} at Epoch {best_epoch+1}")
    run.best_map = best_map
    run.best_epoch = best_epoch
    run.history = history
    return run

def main(args: argparse.Namespace) -> None:
    exp = setup_experiment(args.config, args)
    data = setup_data(exp)
    run = setup_model(exp, data)
    run = train(exp, data, run)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    
    # Sweep overrides
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--project", type=str, help="Override wandb project name")
    parser.add_argument("--name", type=str, help="Override experiment name")
    
    # Data split arguments
    parser.add_argument("--mode", type=str, choices=["full", "search"], help="Training mode: 'full' (train on all train, eval on val) or 'search' (train on sub-train, eval on dev)")
    parser.add_argument("--seed", type=int, help="Random seed for data splitting")
    parser.add_argument("--split_ratio", type=float, help="Ratio of training data to keep in 'search' mode")
    
    # Sweep arguments (accepting string 'true'/'false' or int 0/1)
    parser.add_argument("--train_backbone", type=str, help="Override train_backbone (true/false)")
    parser.add_argument("--augment", type=str, help="Override augment (true/false)")

    args = parser.parse_args()
    main(args)
