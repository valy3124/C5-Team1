import os
import sys
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# For DETR custom NMS
from torchvision.ops import nms
from transformers import DetrForObjectDetection, DetrImageProcessor

# Project imports
from src.datasets import KITTIMOTS
from src.fine_tune.utils import get_transforms, KITTIMOTSToTorchvision, ApplyAlbumentations
from src.models.faster_rcnn import FasterRCNNModel

# Class colors: 1=Car (blue), 2=Pedestrian (red) -> Actually in eval_nms_impact it is: {1: "red", 2: "blue"}
CLASS_COLORS = {1: "red", 2: "blue"}

def load_dataset(root="~/mcv/datasets/C5/KITTI-MOTS/", split="dev"):
    """
    Loads the KITTI-MOTS dev dataset.
    Returns:
        base_dataset: Raw dataset for Ground Truth access and PIL Images
        val_dataset: Augmented/transformed dataset for inference
    """
    base_dataset = KITTIMOTS(root=root, split=split, seed=42, split_ratio=0.8)
    val_dataset = ApplyAlbumentations(
        KITTIMOTSToTorchvision(base_dataset),
        tf=get_transforms(is_train=False)
    )
    return base_dataset, val_dataset

def load_faster_rcnn(exp_dir, device="cuda:0"):
    """Loads a fine-tuned Faster R-CNN model from an experiment directory."""
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    freeze_strategy = config.get('training', {}).get('freeze_strategy', 1)
    num_classes = config.get('model', {}).get('num_classes', 3)
    
    model = FasterRCNNModel(weights=None, device=device)
    model.prepare_finetune(num_classes=num_classes, freeze_strategy=freeze_strategy)
    
    weights_path = os.path.join(exp_dir, "best_model.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_detr(exp_dir, device="cuda:0"):
    """Loads a fine-tuned DETR model from an experiment directory."""
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    weights = config["model"].get("weights", "facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained(weights)
    
    # Standard KITTI-MOTS classes
    classes = ['background', 'Car', 'Pedestrian']
    id2label = {i: label for i, label in enumerate(classes)}
    label2id = {label: i for i, label in enumerate(classes)}
    
    model = DetrForObjectDetection.from_pretrained(
        weights,
        num_labels=len(classes),
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id,
    )
    
    weights_path = os.path.join(exp_dir, "best_model.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, processor

def predict_and_nms(model, image, iou_thresholds, model_type="faster_rcnn", processor=None, device="cuda:0"):
    """
    Returns a dict {iou_thresh: {'boxes': ..., 'scores': ..., 'labels': ...}}
    For NMS=None, no NMS (or a very high threshold like 1.0) is applied.
    """
    predictions_dict = {}
    
    with torch.no_grad():
        if model_type == "faster_rcnn":
            original_nms_thresh = model.model.roi_heads.nms_thresh
            for iou in iou_thresholds:
                # 1.0 means no boxes will be suppressed by NMS
                model.model.roi_heads.nms_thresh = 1.0 if iou is None else iou
                
                pred = model.predict(image)
                predictions_dict[iou] = {
                    'boxes': pred['bboxes_xyxy'],
                    'scores': pred['scores'],
                    'labels': pred['category_ids']
                }
            # Restore original state
            model.model.roi_heads.nms_thresh = original_nms_thresh
            
        elif model_type == "detr":
            # Image needs to be processed
            if hasattr(image, "numpy"):
                image_arr = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_arr = image
                
            inputs = processor(images=image_arr, return_tensors="pt", do_rescale=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            # Identify image dimensions
            if isinstance(image_arr, np.ndarray):
                h, w = image_arr.shape[:2]
            else: # PIL Image
                w, h = image_arr.size
                
            target_sizes = torch.tensor([[h, w]]).to(device)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)[0]
            
            boxes = results["boxes"]
            scores = results["scores"]
            labels = results["labels"]
            
            for iou in iou_thresholds:
                if iou is None:
                    predictions_dict[iou] = {
                        'boxes': boxes.cpu().numpy(),
                        'scores': scores.cpu().numpy(),
                        'labels': labels.cpu().numpy()
                    }
                else:
                    keep = nms(boxes, scores, iou)
                    predictions_dict[iou] = {
                        'boxes': boxes[keep].cpu().numpy(),
                        'scores': scores[keep].cpu().numpy(),
                        'labels': labels[keep].cpu().numpy()
                    }
    return predictions_dict

def draw_boxes(ax, boxes, labels, scores=None, color_map=CLASS_COLORS):
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        color = color_map.get(int(label), "yellow")
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        if scores is not None:
            ax.text(x1, y1 - 2, f"{scores[i]:.2f}", color=color, fontsize=12, va='bottom', fontweight='bold')

def visualize_qualitative_results(image, gt_anns, predictions_dict, conf_threshold=0.5):
    num_subplots = 1 + len(predictions_dict)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 4 * num_subplots))
    
    if num_subplots == 1:
        axes = [axes]
        
    # display GT
    if hasattr(image, "numpy"):
        img_display = image.permute(1, 2, 0).cpu().numpy()
    else:
        img_display = image
        
    axes[0].imshow(img_display)
    gt_boxes  = np.array([ann.bbox_xyxy for ann in gt_anns])
    gt_labels = np.array([ann.class_id   for ann in gt_anns])
    if len(gt_boxes) > 0:
        draw_boxes(axes[0], gt_boxes, gt_labels)
    axes[0].set_title(f"Ground Truth ({len(gt_boxes)} boxes)", fontsize=16, fontweight='bold')
    axes[0].axis("off")
    
    # Sort IOUs: small to big, then None (no NMS) at the end
    valid_ious = [k for k in predictions_dict.keys() if k is not None]
    sorted_ious = sorted(valid_ious)
    if None in predictions_dict:
        sorted_ious.append(None)
        
    # display predicions
    for ax_idx, iou_thresh in enumerate(sorted_ious, start=1):
        pred = predictions_dict[iou_thresh]
        axes[ax_idx].imshow(img_display)

        valid = pred['scores'] >= conf_threshold
        boxes  = pred['boxes'][valid]
        labels = pred['labels'][valid]
        scores = pred['scores'][valid]

        if len(boxes) > 0:
            draw_boxes(axes[ax_idx], boxes, labels, scores=scores)

        title = "No NMS" if iou_thresh is None else f"NMS={iou_thresh}"
        axes[ax_idx].set_title(f"{title} ({valid.sum()} boxes)", fontsize=16, fontweight='bold')
        axes[ax_idx].axis("off")

    plt.tight_layout()
    plt.show()


import torch
import json
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.datasets import KITTIMOTS
from src.fine_tune.utils import get_transforms, collate_fn, KITTIMOTSToTorchvision, ApplyAlbumentations
from src.fine_tune.fine_tune_faster_rcnn import evaluate
from src.inference.evaluation import CocoMetrics
from src.models.faster_rcnn import FasterRCNNModel
from transformers import DetrForObjectDetection, DetrImageProcessor

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MockExp:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

class MockRun:
    def __init__(self, m):
        self.model = m

def run_quantitative_eval(model_type, exp_dir, root_dir="/export/home/group01/mcv/datasets/C5/KITTI-MOTS/", iou_thresholds=[0.3, 0.5, 0.7, 0.9], split="dev"):
    """
    Runs quantitative NMS evaluation (mAP) for a given model across multiple IOU thresholds.
    
    Args:
        model_type (str): 'faster_rcnn' or 'detr'
        exp_dir (str): Path to the experiment directory containing best_model.pth or DETR weights
        root_dir (str): Path to KITTI-MOTS dataset base directory
        iou_thresholds (list): List of NMS thresholds to evaluate
        split (str): Dataset split to evaluate on (default: 'dev')
        
    Returns:
        dict: A dictionary mapping IOU thresholds to their corresponding coco metrics dictionaries.
    """
    import yaml
    
    exp_dir = Path(exp_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load config if faster_rcnn, detrimental for DETR but we try
    cfg = {}
    config_path = exp_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

    # 1. Load Dataset & DataLoader
    logging.info(f"Loading {split} Dataset...")
    val_base = KITTIMOTS(root=root_dir, split=split, seed=42, split_ratio=0.8)
    val_ds   = ApplyAlbumentations(
        KITTIMOTSToTorchvision(val_base),
        tf=get_transforms(is_train=False)
    )
    val_loader  = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    metrics_obj = CocoMetrics(root=root_dir, split=split, ann_source="txt", seed=42, split_ratio=0.8)
    
    # 2. Setup Model specifics
    num_classes = 3 # background (0) + Car (1) + Pedestrian (2) for Faster RCNN
    
    if model_type == 'faster_rcnn':
        weights_path = exp_dir / "best_model.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found at {weights_path}")
        freeze_strategy = cfg.get('training', {}).get('freeze_strategy', 1)
        
    elif model_type == 'detr':
        # Create an early instance to see if we can do this without full huggingface loop,
        # but evaluation loop expects `evaluate(exp, run, val_loader, metrics_obj)`
        # which uses `run.model.predict(...)`. 
        # The Custom DETR wrapper from fine_tune_detr.py isn't easily exportable here 
        # without bringing in all of fine_tune_detr.py's internal classes.
        # We will build a custom small evaluation loop just for calculating metrics.
        weights = cfg.get("model", {}).get("weights", "facebook/detr-resnet-50")
        processor = DetrImageProcessor.from_pretrained(weights, revision="no_timm")
        
        # Standard KITTI-MOTS classes
        classes = ['background', 'Car', 'Pedestrian']
        id2label = {i: label for i, label in enumerate(classes)}
        label2id = {label: i for i, label in enumerate(classes)}
        
        model = DetrForObjectDetection.from_pretrained(
            weights,
            num_labels=len(classes),
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
        )
        
        weights_path = exp_dir / "best_model.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"DETR Weights not found at {weights_path}")
            
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    metrics_results = {}
    
    for iou in iou_thresholds:
        logging.info(f"\n=================================\nEvaluating NMS Threshold: {iou}\n=================================")
        
        if model_type == 'faster_rcnn':
            # Faster R-CNN re-init with new IOU (1.0 means no NMS suppression)
            effective_iou = 1.0 if iou is None else iou
            model = FasterRCNNModel(weights=None, device=str(device), iou=effective_iou)
            model.prepare_finetune(num_classes=num_classes, freeze_strategy=freeze_strategy)
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.eval()
            
            exp = MockExp(cfg, device)
            run = MockRun(m=model)
            
            logging.info("Running evaluation...")
            eval_result = evaluate(exp, run, val_loader, metrics_obj)
            map_score   = eval_result.metrics.get('overall/AP', 0.0)
            logging.info(f"-> mAP for NMS {iou}: {map_score:.4f}")
            metrics_results[iou] = eval_result.metrics
            
        elif model_type == 'detr':
            from torchvision.ops import nms
            
            # Custom evaluation loop for DETR
            coco_results = []
            
            logging.info("Running inference on dev set...")
            for batch in tqdm(val_loader, desc="Evaluating DETR"):
                images, targets = batch
                
                # Convert list of tensors to list of numpy arrays/PIL for processor
                # But our val_loader gives us preprocessed tensors.
                # DETR pipeline usually expects raw images.
                # Let's use the dataset iteratively instead to get raw images.
                pass
                
            # Due to the complexity of the dataloader differences, we will do a direct loop over validation base
            coco_results = []
            for idx in tqdm(range(len(val_base)), desc=f"Evaluating DETR NMS={iou}"):
                image, anns, meta = val_base[idx]
                image_id = meta["index"]
                
                # Image is either PIL or tensor depending on Albumentations, val_base gives PIL/numpy
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                target_sizes = torch.tensor([image.size[::-1]]).to(device) # (h, w)
                results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)[0]
                
                boxes = results["boxes"]
                scores = results["scores"]
                labels = results["labels"]
                
                # Apply NMS
                if iou is not None:
                    keep = nms(boxes, scores, iou)
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                
                # Convert to COCO format
                # DETR outputs labels corresponding to Model Config. 0 = Person, 1 = Car typically
                # CocoMetrics expects original kitty mappings: 1=Person, 3=Car
                # We map label 0 -> 1, label 1 -> 3
                detr_to_coco = {0: 1, 1: 3}
                
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    cat_id = detr_to_coco.get(label.item(), 1)
                    
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, x2-x1, y2-y1], # xywh
                        "score": score.item()
                    })
                    
            # Compute metrics
            logging.info("Computing COCO metrics...")
            coco_dt = metrics_obj.coco_gt.loadRes(coco_results)
            metrics = metrics_obj.compute_metrics(coco_dt)
            map_score = metrics.get('overall/AP', 0.0)
            logging.info(f"-> mAP for NMS {iou}: {map_score:.4f}")
            metrics_results[iou] = metrics

    return metrics_results

def print_metrics_summary(metrics_results):
    """
    Prints a nicely formatted summary table of the evaluation results.
    """
    print("\n{:>6} | {:>10} | {:>10} | {:>10}".format("NMS", "AP", "AP_50", "AP_75"))
    print("-" * 44)
    for iou, m in metrics_results.items():
        iou_str = str(iou) if iou is not None else "None"
        print("{:>6} | {:>10.4f} | {:>10.4f} | {:>10.4f}".format(
            iou_str,
            m.get('overall/AP', 0),
            m.get('overall/AP_50', 0),
            m.get('overall/AP_75', 0),
        ))
