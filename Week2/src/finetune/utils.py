"""
utils.py — Shared utilities for fine-tuning SAM.
Contains standard dataset adapters, augmentations, and experiment setup logic.
"""

import os
import sys
import argparse
import yaml
import torch
import wandb
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Path bootstrap
def bootstrap_paths():
    """Add src/ and Week2/ to sys.path so we can import project modules."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir     = os.path.dirname(current_dir)
    week2_dir   = os.path.dirname(src_dir)
    for p in (src_dir, week2_dir):
        if p not in sys.path:
            sys.path.append(p)

bootstrap_paths()

# Data Containers
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
    label_mapping:      Dict[int, int] = None

@dataclass
class Run:
    model:      Any
    optimizer:  Any
    history:    Dict[str, list]
    processor:  Any = None        # Used by SAM
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

# Utility Helpers
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
    parser.add_argument("--aug_strategy", type=str, help="Augmentation strategy name.")
    parser.add_argument("--name_fields", type=str, help="Fields for auto-generated run name.")
    parser.add_argument("--optimizer", type=str, help="Override optimizer.")
    parser.add_argument("--scheduler", type=str, help="Override scheduler name.")
    parser.add_argument("--gradient_clipping", type=float, help="Override gradient clipping max norm.")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only, skip training.")
    return parser

# Setup Functions
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
    
    if getattr(args, "aug_strategy", None) is not None: training["aug_strategy"]     = str(args.aug_strategy)
    if getattr(args, "optimizer", None) is not None: training["optimizer"] = str(args.optimizer)
    if getattr(args, "scheduler", None) is not None: training.setdefault("scheduler", {})["name"] = str(args.scheduler)
    if getattr(args, "gradient_clipping", None) is not None: training["gradient_clipping"] = float(args.gradient_clipping)

    cfg.setdefault("data", {})
    cfg["data"]["mode"]        = args.mode        or cfg["data"].get("mode", "full")
    cfg["data"]["seed"]        = args.seed        or cfg["data"].get("seed", 42)
    cfg["data"]["split_ratio"] = args.split_ratio or cfg["data"].get("split_ratio", 0.8)

    if not args.name:
        model_name = cfg["model"].get("name", "model")
        if model_name == "sam_vit_b":
            model_name = "sam"
        parts = [model_name]
        
        aug = training.get("aug_strategy", "base")
        parts.append(str(aug))
        
        prompt_type = training.get("prompt_type", "bbox")
        parts.append(prompt_type)
        
        if "lr" in training:
            parts.append(f"LR{training['lr']:.1e}")
            
        wd = training.get("weight_decay", 0)
        if wd != 0:
            parts.append(f"WD{wd:.1e}")
            
        if "optimizer" in training:
            parts.append(f"Opt_{training['optimizer']}")
            
        if "scheduler" in training and training["scheduler"].get("name", "none") != "none":
            parts.append(f"Sch_{training['scheduler']['name']}")
            
        if "gradient_clipping" in training:
            parts.append(f"GC_{training['gradient_clipping']:.1f}")

        is_sweep = bool(os.environ.get("WANDB_SWEEP_ID"))
        if is_sweep:
            parts = ["sam", str(training.get("aug_strategy", "base")), prompt_type]
            
        cfg["experiment_name"] = "_".join(parts)
        print(f"Auto-generated experiment name: {cfg['experiment_name']}")

    project_name = cfg.get("project", "C5-Team1-Week2-SAM")
    if wandb.run is None:
        wandb.init(project=project_name, name=cfg.get("experiment_name", "run"), config=cfg)
    else:
        # Update config if already initialized
        wandb.config.update(cfg, allow_val_change=True)

    is_sweep = bool(os.environ.get("WANDB_SWEEP_ID"))
    model_dir  = "sam_aug" if is_sweep else cfg["model"].get("name", "unknown_model")
    
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
