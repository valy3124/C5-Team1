"""
fine_tune_yolo.py
Fine-tunes Ultralytics YOLO on KITTI-MOTS using native Ultralytics intrinsic augmentations
and custom callbacks for detailed COCO evaluation and W&B logging.
"""

import os
import sys
import argparse
import json
import shutil
import yaml
import torch
import wandb
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from PIL import Image

from src.datasets import KITTIMOTS
from src.inference.evaluation import CocoMetrics
from src.models.yolo import UltralyticsYOLO

# Setup paths
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
week1_dir = src_dir.parent

for p in (str(src_dir), str(week1_dir)):
    if p not in sys.path:
        sys.path.append(p)

# Dataset Definitions & Mappings

# KITTI-MOTS: 1=Car, 2=Pedestrian -> YOLO: 0=Car, 1=Pedestrian
_KITTI_TO_YOLO_CLS: Dict[int, int] = {1: 0, 2: 1}
YOLO_CLASS_NAMES: List[str] = ["Car", "Pedestrian"]

# YOLO output -> COCO standard IDs: 0(Car)=3, 1(Pedestrian)=1
_YOLO_TO_COCO_LABEL: Dict[int, int] = {0: 3, 1: 1}


# Data Containers
@dataclass
class Exp:
    """Experiment configuration and environment state."""
    cfg: Dict[str, Any]
    device: torch.device
    output_dir: Path
    best_model_path: Path

@dataclass
class Data:
    """Dataset metadata and evaluation helpers."""
    classes: List[str]
    yolo_dataset_dir: Path
    data_yaml_path: Path
    train_split: str
    val_split: str
    raw_val_ds: Any
    train_coco_metrics: Any = None
    val_coco_metrics: Any = None

@dataclass
class Run:
    """Model instance and training history."""
    model: Any
    history: Dict[str, list] = field(default_factory=lambda: {
        "train_box_loss": [], "train_cls_loss": [], "train_dfl_loss": [],
        "val_box_loss": [], "val_cls_loss": [], "val_dfl_loss": [],
        "val_map": [], "val_map50": [],
    })
    best_map: float = 0.0
    best_epoch: int = 0

@dataclass
class Eval:
    """Outputs produced by a single COCO evaluation pass."""
    predictions: List[Dict]
    metrics: Dict[str, float]

# Utility Helpers & Augmentation Profiles

def _xyxy_to_xywh(box) -> List[float]:
    """Convert [x1, y1, x2, y2] to COCO [x, y, w, h] format."""
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, x2 - x1, y2 - y1]

def get_augmentation_profile(strategy: str) -> Dict[str, float]:
    """
    Returns a dictionary of Ultralytics intrinsic augmentation parameters.
    Add or modify profiles here to easily control augmentations via CLI.
    """
    profiles = {
        "none": {
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
            "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0,
            "perspective": 0.0, "flipud": 0.0, "fliplr": 0.0,
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0
        },
        "base": {  # Standard YOLO default
            "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
            "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
            "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
            "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0
        },
        "heavy": { # Aggressive geometry and color changes
            "hsv_h": 0.1, "hsv_s": 0.9, "hsv_v": 0.9,
            "degrees": 10.0, "translate": 0.2, "scale": 0.7, "shear": 2.0,
            "perspective": 0.001, "flipud": 0.0, "fliplr": 0.5,
            "mosaic": 1.0, "mixup": 0.2, "copy_paste": 0.1
        },
        "color_only": { # Isolate color jittering, no geometry
            "hsv_h": 0.1, "hsv_s": 0.9, "hsv_v": 0.9,
            "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0,
            "perspective": 0.0, "flipud": 0.0, "fliplr": 0.0,
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0
        },
        "geometry_only": { 
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
            "degrees": 10.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
            "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0
        },
        "composite_only": { 
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
            "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0,
            "perspective": 0.0, "flipud": 0.0, "fliplr": 0.0,
            "mosaic": 1.0, "mixup": 0.2, "copy_paste": 0.0
        }
    }
    
    if strategy not in profiles:
        print(f"Warning: Augmentation strategy '{strategy}' not found. Defaulting to 'base'.")
        return profiles["base"]
        
    return profiles[strategy]

# Data Export Functions

def export_yolo_dataset(raw_ds, split_name: str, output_root: Path) -> None:
    """Exports a raw KITTI-MOTS split to the YOLO format on disk."""
    img_dir = output_root / "images" / split_name
    lbl_dir = output_root / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(raw_ds)), desc=f"Export {split_name}", ascii=True):
        img_pil, raw_anns, _ = raw_ds[idx]
        w_img, h_img = img_pil.size

        img_pil.save(img_dir / f"{idx:06d}.jpg", quality=95)

        rows = []
        for ann in raw_anns:
            cls_id = int(getattr(ann, "class_id", -1))
            yolo_cls = _KITTI_TO_YOLO_CLS.get(cls_id, -1)
            box = getattr(ann, "bbox_xyxy", None)
            
            if yolo_cls < 0 or box is None:
                continue

            x1, y1, x2, y2 = map(float, box)
            x2 += 1; y2 += 1

            bw, bh = (x2 - x1) / w_img, (y2 - y1) / h_img
            if bw <= 0 or bh <= 0:
                continue

            cx, cy = (x1 + x2) / 2.0 / w_img, (y1 + y2) / 2.0 / h_img
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            bw, bh = min(bw, 1.0), min(bh, 1.0)
            
            rows.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(lbl_dir / f"{idx:06d}.txt", "w") as fh:
            fh.write("\n".join(rows))

def write_yolo_data_yaml(dataset_root: Path) -> Path:
    """Generates the data.yaml required by Ultralytics YOLO."""
    yaml_path = dataset_root / "data.yaml"
    data = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(YOLO_CLASS_NAMES),
        "names": YOLO_CLASS_NAMES,
    }
    with open(yaml_path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False)
    print(f"YOLO data.yaml → {yaml_path}")
    return yaml_path

# Setup Functions

def setup_experiment(config_path: str, args: argparse.Namespace) -> Exp:
    """Loads config, applies CLI overrides, and sets up paths/W&B."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    training = cfg.setdefault("training", {})
    
    # CLI Overrides
    if args.epochs: training["epochs"] = args.epochs
    if args.batch_size: training["batch_size"] = args.batch_size
    if args.lr: training["lr"] = args.lr
    if args.imgsz: training["imgsz"] = args.imgsz
    if args.project: cfg["project"] = args.project
    if args.name: cfg["experiment_name"] = args.name
    if args.freeze is not None: training["freeze"] = int(args.freeze)
    if args.aug_strategy is not None: training["aug_strategy"] = str(args.aug_strategy)

    #--------
    # Dynamic Stages for Sweep
    if getattr(args, "epochs_head", 0) > 0 or getattr(args, "epochs_neck", 0) > 0 or getattr(args, "epochs_backbone", 0) > 0:
        dynamic_stages = []
        if getattr(args, "epochs_head", 0) > 0:
            dynamic_stages.append({
                "name": "head_only", "epochs": args.epochs_head, 
                "lr": getattr(args, "lr_head", 0.001), "freeze": 22
            })
        if getattr(args, "epochs_neck", 0) > 0:
            dynamic_stages.append({
                "name": "neck_head", "epochs": args.epochs_neck, 
                "lr": getattr(args, "lr_neck", 0.0001), "freeze": 10
            })
        if getattr(args, "epochs_backbone", 0) > 0:
            dynamic_stages.append({
                "name": "full_ft", "epochs": args.epochs_backbone, 
                "lr": getattr(args, "lr_backbone", 0.00001), "freeze": 0
            })
        if dynamic_stages:
            training["stages"] = dynamic_stages
    #--------
    cfg.setdefault("data", {})
    cfg["data"]["mode"] = args.mode or cfg["data"].get("mode", "full")
    cfg["data"]["seed"] = args.seed or cfg["data"].get("seed", 42)
    cfg["data"]["split_ratio"] = args.split_ratio or cfg["data"].get("split_ratio", 0.8)

    # W&B and Paths
    wandb.init(
        project=cfg.get("project", "kitti-mots-yolo"),
        name=cfg.get("experiment_name", "yolo_run"),
        config=cfg,
    )

    output_dir = Path(cfg.get("output_dir", "results")) / cfg.get("model", {}).get("name", "yolo") / f"{cfg.get('experiment_name', 'exp')}_{wandb.run.id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    return Exp(cfg, device, output_dir, output_dir / "best_model.pt")

def setup_data(exp: Exp) -> Data:
    """Prepares datasets and CocoMetrics evaluators."""
    cfg = exp.cfg
    root = cfg["data"]["root"]
    mode = cfg["data"]["mode"]
    seed = cfg["data"]["seed"]
    split_ratio = cfg["data"]["split_ratio"]

    if mode == "full":
        train_split, val_split = "train_full", "validation"
    elif mode == "search":
        train_split, val_split = "train", "dev"
    else:
        raise ValueError("Unknown data.mode. Valid options: full, search.")

    raw_train_ds = KITTIMOTS(root, split=train_split, ann_source="txt", seed=seed, split_ratio=split_ratio)
    raw_val_ds = KITTIMOTS(root, split=val_split, ann_source="txt", seed=seed, split_ratio=split_ratio)

    yolo_dataset_dir = exp.output_dir / "yolo_dataset"
    train_img_dir = yolo_dataset_dir / "images" / "train"
    val_img_dir = yolo_dataset_dir / "images" / "val"

    n_train = len(list(train_img_dir.glob("*.jpg"))) if train_img_dir.is_dir() else 0
    n_val = len(list(val_img_dir.glob("*.jpg"))) if val_img_dir.is_dir() else 0

    if n_train == len(raw_train_ds) and n_val == len(raw_val_ds):
        print("YOLO dataset already exported. Skipping.")
    else:
        export_yolo_dataset(raw_train_ds, "train", yolo_dataset_dir)
        export_yolo_dataset(raw_val_ds, "val", yolo_dataset_dir)

    data_yaml_path = write_yolo_data_yaml(yolo_dataset_dir)

    return Data(
        classes=YOLO_CLASS_NAMES,
        yolo_dataset_dir=yolo_dataset_dir,
        data_yaml_path=data_yaml_path,
        train_split=train_split,
        val_split=val_split,
        raw_val_ds=raw_val_ds,
        train_coco_metrics=CocoMetrics(root=root, split=train_split, ann_source="txt", seed=seed, split_ratio=split_ratio),
        val_coco_metrics=CocoMetrics(root=root, split=val_split, ann_source="txt", seed=seed, split_ratio=split_ratio),
    )

def setup_model(exp: Exp) -> Run:
    """Initializes the Ultralytics wrapper."""
    cfg = exp.cfg
    if cfg["model"].get("name", "yolo") != "yolo":
        raise ValueError("This script only supports 'yolo'.")

    device_str = "0" if exp.device.type == "cuda" else "cpu"
    wrapper = UltralyticsYOLO(
        weights=cfg["model"].get("weights", "yolov10b.pt"),
        conf=cfg["training"].get("conf", 0),
        iou=cfg["training"].get("nms_iou_threshold", 0.5),
        device=device_str,
        map_coco=False,
    )
    return Run(model=wrapper)

# Evaluation & Logging

def evaluate(exp: Exp, run: Run, data: Data, use_val: bool = True) -> Eval:
    """Evaluates model using COCO metrics over raw datasets."""
    model = run.model
    raw_ds = data.raw_val_ds if use_val else None
    metrics_obj = data.val_coco_metrics if use_val else data.train_coco_metrics
    batch_size = 8 

    if model is None or raw_ds is None:
        return Eval(predictions=[], metrics={"overall/AP": 0.0})

    coco_dt_list: List[Dict] = []

    for start in tqdm(range(0, len(raw_ds), batch_size), desc="COCO eval", ascii=True):
        batch = [raw_ds[i] for i in range(start, min(start + batch_size, len(raw_ds)))]
        pil_imgs = [item[0] for item in batch]
        preds, _ = model.predict(pil_imgs)

        for pred, (_, _, meta) in zip(preds, batch):
            image_id = int(meta.get("index", start))
            for box, cat, score in zip(pred["bboxes_xyxy"], pred["category_ids"], pred["scores"]):
                cat_id = int(cat)
                if cat_id not in _YOLO_TO_COCO_LABEL:
                    continue
                coco_dt_list.append({
                    "image_id": image_id,
                    "category_id": _YOLO_TO_COCO_LABEL[cat_id],
                    "bbox": _xyxy_to_xywh(box),
                    "score": float(score),
                })

    if not coco_dt_list:
        return Eval(predictions=[], metrics={"overall/AP": 0.0})

    coco_dt = metrics_obj.coco_gt.loadRes(coco_dt_list)
    metrics_result = metrics_obj.compute_metrics(coco_dt)
    return Eval(predictions=coco_dt_list, metrics=metrics_result)

def log_predictions_to_wandb(exp: Exp, run: Run, data: Data, epoch: int, max_images: int = 4) -> None:
    """Logs inference results vs ground truth on W&B."""
    if run.model is None: return

    CLASS_NAMES = {1: "Car", 2: "Pedestrian"}
    raw_ds = data.raw_val_ds
    step = max(1, len(raw_ds) // max_images)
    indices = list(range(0, len(raw_ds), step))[:max_images]
    logged_images = []

    for idx in indices:
        img_pil, raw_anns, _ = raw_ds[idx]
        pred_list, _ = run.model.predict([img_pil])
        pred = pred_list[0]

        wandb_pred = [{
            "position": {"minX": float(b[0]), "minY": float(b[1]), "maxX": float(b[2]), "maxY": float(b[3])},
            "class_id": int(c),
            "box_caption": f"{CLASS_NAMES.get(int(c), '?')} {s:.2f}",
            "scores": {"score": float(s)},
            "domain": "pixel",
        } for b, c, s in zip(pred["bboxes_xyxy"], pred["category_ids"], pred["scores"]) if s >= 0.25]

        wandb_gt = []
        for ann in raw_anns:
            cls_kitti = int(getattr(ann, "class_id", -1))
            if getattr(ann, "bbox_xyxy", None) is None: continue
            x1, y1, x2, y2 = map(float, ann.bbox_xyxy)
            wandb_gt.append({
                "position": {"minX": x1, "minY": y1, "maxX": x2 + 1, "maxY": y2 + 1},
                "class_id": cls_kitti,
                "box_caption": f"{CLASS_NAMES.get(cls_kitti, '?')} (GT)",
                "domain": "pixel",
            })

        logged_images.append(wandb.Image(np.array(img_pil), boxes={
            "predictions": {"box_data": wandb_pred, "class_labels": CLASS_NAMES},
            "ground_truth": {"box_data": wandb_gt, "class_labels": CLASS_NAMES},
        }))

    if logged_images:
        wandb.log({"val_predictions": logged_images, "epoch": epoch}, commit=False)

# Training

def train(exp: Exp, data: Data, run: Run) -> Run:
    cfg = exp.cfg
    training = cfg["training"]
    stages = training.get("stages", None)
    if not stages:
        stages = [{
            "name": "run",
            "epochs": training.get("epochs", 40),
            "lr": training.get("lr", 1e-4),
            "freeze": training.get("freeze", 0)
        }]

    batch_size = training.get("batch_size", 16)
    aug_strategy = training.get("aug_strategy", "base")
    aug_kwargs = get_augmentation_profile(aug_strategy)
    imgsz = training.get("imgsz", 640)
    ul_project_dir = exp.output_dir / "ultralytics_runs"
    csv_path = exp.output_dir / "metrics_history.csv"

    state = {
        "overall_best_map": 0.0,
        "global_epoch_offset": 0,
        "stage_name": "",
        "num_epochs": 0,
        "best_map_stage": 0.0,
        "best_epoch_stage": 0
    }

    ul_project_dir = exp.output_dir / "ultralytics_runs"
    csv_path = exp.output_dir / "metrics_history.csv"

    state = {
        "overall_best_map": 0.0,
        "global_epoch_offset": 0,
        "stage_name": "",
        "num_epochs": 0,
        "best_map_stage": 0.0,
        "best_epoch_stage": 0
    }

    def on_fit_epoch_end(trainer):
        local_epoch = trainer.epoch + 1
        global_epoch = state["global_epoch_offset"] + local_epoch
        
        # Log Augmented Data
        if global_epoch == 1 and wandb.run is not None:
            batch_images = list(Path(trainer.save_dir).glob("train_batch*.jpg"))
            if batch_images:
                wandb.log({"augmented_train_batches": [wandb.Image(str(img)) for img in batch_images]}, step=global_epoch, commit=False)

        metrics_ul = trainer.metrics or {}
        val_box = float(metrics_ul.get("val/box_loss", 0.0))
        val_cls = float(metrics_ul.get("val/cls_loss", 0.0))
        val_dfl = float(metrics_ul.get("val/dfl_loss", 0.0))
        val_map50 = float(metrics_ul.get("metrics/mAP50(B)", 0.0))

        ul_last = Path(trainer.save_dir) / "weights" / "last.pt"
        if not ul_last.is_file():
            print("Warning: last.pt not found, using current run.model (may break training)")
            temp_model = run.model
        else:
            device_str = "0" if exp.device.type == "cuda" else "cpu"
            temp_model = UltralyticsYOLO(
                weights=str(ul_last), conf=run.model.conf, iou=run.model.iou, device=device_str, map_coco=False
            )

        class TempRun:
            model = temp_model

        temp_run = TempRun()
        coco_result = evaluate(exp, temp_run, data, use_val=True)
        val_map_coco = coco_result.metrics.get("overall/AP", 0.0)

        print(f"Stage '{state['stage_name']}' | Epoch {local_epoch}/{state['num_epochs']} (Global {global_epoch}) | COCO AP: {val_map_coco:.4f}")

        if val_map_coco > state["best_map_stage"]:
            state["best_map_stage"] = val_map_coco
            state["best_epoch_stage"] = local_epoch

        if val_map_coco > state["overall_best_map"]:
            state["overall_best_map"] = val_map_coco
            run.best_map = state["overall_best_map"]
            run.best_epoch = global_epoch
            if ul_last.is_file():
                shutil.copy2(str(ul_last), exp.best_model_path)
            with open(exp.output_dir / "best_metrics.json", "w") as fh:
                json.dump(coco_result.metrics, fh, indent=4)

        if local_epoch % training.get("log_images_every", 5) == 0:
            log_predictions_to_wandb(exp, temp_run, data, epoch=global_epoch)

        # Train Loss Logging
        if hasattr(trainer, "tloss") and trainer.tloss is not None:
            train_loss = trainer.tloss.sum().item()
        elif hasattr(trainer, "loss_items") and trainer.loss_items is not None:
            train_loss = trainer.loss_items.sum().item()
        else:
            train_loss = 0.0
        val_loss = val_box + val_cls + val_dfl

        wandb_log = {
            "epoch": global_epoch,
            "stage_name": state["stage_name"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_coco_ap": val_map_coco,
            "best_map": state["overall_best_map"],
            "best_epoch": run.best_epoch,
        }
        wandb_log.update({f"val_{k}": v for k, v in coco_result.metrics.items()})
        wandb.log(wandb_log)

        write_header = not csv_path.is_file()
        with open(csv_path, "a") as fh:
            if write_header:
                fh.write("global_epoch,stage,val_box_loss,val_cls_loss,val_map50,val_coco_ap\n")
            fh.write(f"{global_epoch},{state['stage_name']},{val_box:.6f},{val_cls:.6f},{val_map50:.6f},{val_map_coco:.6f}\n")

    def on_train_end(trainer):
        print(f"Stage '{state['stage_name']}' complete. Stage best COCO AP: {state['best_map_stage']:.4f} at epoch {state['best_epoch_stage']}.")

    # Training Stages
    for stage_idx, stage in enumerate(stages):
        state["stage_name"] = stage.get("name", f"stage_{stage_idx+1}")
        state["num_epochs"] = stage.get("epochs", 10)
        state["best_map_stage"] = 0.0
        state["best_epoch_stage"] = 0
        
        lr = stage.get("lr", 1e-4)
        freeze = stage.get("freeze", 0)
        run.model.model.callbacks["on_fit_epoch_end"] = [on_fit_epoch_end]
        run.model.model.callbacks["on_train_end"] = [on_train_end]

        
        print(f"Starting Stage: {state['stage_name']} | epochs: {state['num_epochs']} | lr: {lr} | freeze: {freeze} | aug: {aug_strategy}")

        # SOnly applied to the first stage
        if stage_idx == 0:
            warmup_e = max(0.0, min(5.0, state["num_epochs"] * 0.1))
        else:
            warmup_e = 0.0
        if "model" not in run.model.model.overrides:
            run.model.model.overrides["model"] = getattr(run.model.model, "ckpt_path", exp.cfg["model"].get("weights", "yolov10b.pt"))

        run.model.model.train(
            data=str(data.data_yaml_path),
            epochs=state["num_epochs"],
            batch=batch_size,
            imgsz=imgsz,
            device="0" if exp.device.type == "cuda" else "cpu",
            lr0=lr,
            cos_lr=True,
            optimizer="AdamW",
            warmup_epochs=warmup_e,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            freeze=freeze,
            project=str(ul_project_dir),
            name=state["stage_name"],
            exist_ok=True,
            save=True,
            verbose=True,
            **aug_kwargs 
        )
        
        state["global_epoch_offset"] += state["num_epochs"]

        # Load best weights from previous stage
        if stage_idx < len(stages) - 1:
            ul_best_relative = Path(ul_project_dir) / state["stage_name"] / "weights" / "best.pt"
            ul_best_runs = Path("runs/detect") / ul_best_relative
            
            ul_best = ul_best_runs if ul_best_runs.is_file() else ul_best_relative

            if ul_best.is_file():
                print(f"Cargando los mejores pesos de {state['stage_name']} desde {ul_best} para la siguiente etapa...")
                device_str = "0" if exp.device.type == "cuda" else "cpu"
                run.model = UltralyticsYOLO(
                    weights=str(ul_best),
                    conf=run.model.conf, 
                    iou=run.model.iou, 
                    device=device_str, 
                    map_coco=False
                )
                run.model.model.overrides["model"] = str(ul_best.resolve())
            else:
                print(f"Advertencia: best.pt de la etapa {state['stage_name']} no encontrado en {ul_best_relative} ni en {ul_best_runs}.")

    print(f"\nAll stages complete. Best Overall COCO AP: {state['overall_best_map']:.4f} at global epoch {run.best_epoch}.")
    return run

# Entry Point
def main(args: argparse.Namespace) -> None:
    exp = setup_experiment(args.config, args)
    data = setup_data(exp)
    run = setup_model(exp)
    run = train(exp, data, run)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Ultralytics YOLO on KITTI-MOTS.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--epochs", type=int, help="Override training.epochs.")
    parser.add_argument("--batch_size", type=int, help="Override training.batch_size.")
    parser.add_argument("--lr", type=float, help="Override training.lr.")
    parser.add_argument("--imgsz", type=int, help="Override training.imgsz.")
    parser.add_argument("--project", type=str, help="Override W&B project name.")
    parser.add_argument("--name", type=str, help="Override W&B run / experiment name.")
    parser.add_argument("--mode", type=str, choices=["full", "search"], help="'full': strict val, 'search': dev split")
    parser.add_argument("--seed", type=int, help="Random seed for data splitting.")
    parser.add_argument("--split_ratio", type=float, help="Train fraction in 'search' mode.")
    parser.add_argument("--freeze", type=int, help="Number of layers to freeze. 0=full training, 22=freeze backbone+neck, etc.")
    parser.add_argument(
        "--aug_strategy", type=str, 
        choices=["none", "base", "heavy", "color_only", "geometry_only", "composite_only"], 
        help="Select an intrinsic Ultralytics augmentation profile."
    )
    
    # Sweep stage overrides
    parser.add_argument("--epochs_head", type=int, default=0, help="Sweep: epochs for head-only stage (freeze 22).")
    parser.add_argument("--lr_head", type=float, default=0.001, help="Sweep: lr for head-only stage.")
    parser.add_argument("--epochs_neck", type=int, default=0, help="Sweep: epochs for neck+head stage (freeze 10).")
    parser.add_argument("--lr_neck", type=float, default=0.0001, help="Sweep: lr for neck+head stage.")
    parser.add_argument("--epochs_backbone", type=int, default=0, help="Sweep: epochs for full finetune (freeze 0).")
    parser.add_argument("--lr_backbone", type=float, default=0.00001, help="Sweep: lr for full finetune stage.")
    
    main(parser.parse_args())