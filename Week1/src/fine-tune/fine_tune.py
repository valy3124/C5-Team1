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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import wandb
from pathlib import Path

# --- 3. Project Imports ---
from models import faster_rcnn
import datasets



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
    test_loader: Any             # DataLoader for the test set


@dataclass
class Run:
    """Container for model, optimization, and training state."""

    model: Any                   # Neural network model instance
    optimizer: Any               # Optimizer instance
    scheduler: Any = None        # Optional scheduler
    history: Dict[str, list]     # Dictionary storing training and test metrics
    best_map: float = 0.0        # Best mAP achieved so far
    best_epoch: int = 0          # Epoch corresponding to best mAP


@dataclass
class Eval:
    """Container for evaluation outputs."""
    predictions: List[Dict]
    metrics: Dict[str, float]

def get_transforms(cfg, is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.2),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        return A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

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
    train_ds_base = datasets.KITTIMOTS(root, split="training", ann_source="txt")
    train_dataset = KITTIMOTSTorchvisionAdapter(train_ds_base, albumentations_tf=get_transforms(cfg, True))

    val_ds_base = datasets.KITTIMOTS(root, split="testing")
    val_dataset = KITTIMOTSTorchvisionAdapter(val_ds_base, albumentations_tf=get_transforms(cfg, False))

    print(f"Data Loaded: {len(train_dataset)} Train, {len(val_dataset)} Test")

    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg['training']['batch_size'], 
                              shuffle=True, collate_fn=collate_fn, 
                              num_workers=cfg['data'].get('num_workers', 4))
    
    test_loader = DataLoader(val_dataset, 
                            batch_size=1, 
                            shuffle=False, collate_fn=collate_fn, 
                            num_workers=cfg['data'].get('num_workers', 4))
    
    classes = ["Background", "Car", "Pedestrian"] # TODO: Check and change this to the actual class names
    return Data(classes, train_loader, test_loader)

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
        model_wrapper = faster_rcnn.FasterRCNNModel(device=str(device))
        model = model_wrapper.model
        
        # --- Manual Surgery to match fine_tune logic ---
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        
        # Replace the head
        num_classes = len(data.classes) # 3: BG, Car, Ped
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Freeze backbone?
        train_backbone = cfg['training'].get('train_backbone', False)
        for p in model.backbone.parameters():
            p.requires_grad = train_backbone
            
        model.to(device)

        # Optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=cfg['training']['lr'])
        
        # Scheduler TODO: modify later
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

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

    history = {'train_loss': [], 'train_map': [], 'test_loss': [], 'test_map': []}
    
    return Run(model, optimizer, history, scheduler, best_map=0.0, best_epoch=0)

def evaluate(exp: Exp, data: Data, run: Run) -> Eval:
    # mAP Logic removed as requested
    print("Evaluation/mAP logic disabled for now.")
    return Eval([], {'map': 0.0})

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
    test_loader = data.test_loader

    model = run.model
    optimizer = run.optimizer
    history = run.history
    best_map = run.best_map
    best_epoch = run.best_epoch

    model_name = cfg['model']['name']
    
    if run.model is None:
        print("Model is None, returning.")
        return run

    print("Starting training...")
    for epoch in tqdm(range(cfg['training']['epochs']), desc="TRAINING THE MODEL"):
        
        model.train()
        epoch_loss = 0
        prog_bar = tqdm(data.train_loader, desc=f"Epoch {epoch+1}")
        
        for images, targets in prog_bar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
        
        train_loss = train_loss / len(train_loader)

        # TODO: compute mAP
        train_map = 0.0

        # Test Loop
        model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                test_loss += losses.item()
        
        test_loss = test_loss / len(test_loader)

        # TODO: compute mAP
        test_map = 0.0

        
        history['train_loss'].append(train_loss)
        history['train_map'].append(train_map)
        history['test_loss'].append(test_loss)
        history['test_map'].append(test_map)
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} | "
              f"Train Loss: {train_loss:.4f} Train mAP: {train_map:.4f} | "
              f"Test Loss: {test_loss:.4f} Test mAP: {test_map:.4f}")
        
        if scheduler:
            scheduler.step()
            
        # Save Best Model based on Test mAP
        if test_map > best_map:
            best_map = test_map
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model: {best_map:.4f} mAP at Epoch {epoch+1}")
        
        wandb.log({
            "epoch": epoch+1, 
            "train_loss": train_loss, 
            "train_map": train_map, 
            "test_loss": test_loss, 
            "test_map": test_map,
            "best_map": best_map,
            "best_epoch": best_epoch,
        })

    print(f"Training finished. Best model: {best_map:.4f} mAP at Epoch {best_epoch+1}")
    torch.save(model.state_dict(), best_model_path)
    run.best_map = best_map
    run.best_epoch = best_epoch
    run.history = history

    return run

def evaluate(exp: Exp, data: Data, run: Run) -> Eval:
    # TODO: implement evaluation of the best model on the test set with COCO metrics
    pass

def main(config_path: str) -> None:
    exp = setup_experiment(config_path)
    data = setup_data(exp)
    run = setup_model(exp, data)
    run = train(exp, data, run)
    eval_ = evaluate(exp, data, run)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
