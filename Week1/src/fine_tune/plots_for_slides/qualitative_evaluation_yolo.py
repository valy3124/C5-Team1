import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from pathlib import Path
from PIL import Image

# Add src to Python Path
BASE_DIR = "/ghome/group01/C5/benet/C5-Team1/Week1"
sys.path.append(BASE_DIR)

from src.datasets import KITTIMOTS
from src.models.yolo import UltralyticsYOLO

# Set paths
RESULTS_DIR = os.path.join(BASE_DIR, "results/yolo/qualitative_evaluation")

# Models info
FINETUNED_WEIGHTS = os.path.join(BASE_DIR, "results/yolo/yolo_sweep_7q7hwud0/best_model.pt")
# config.yaml says yolov10b.pt was used
PRETRAINED_WEIGHTS = "yolov10b.pt"

# Limits for evaluation
LIMIT = 100
MAX_DET = 100

# Dataset class mapping
# Model/GT usually outputs: 1=Car, 2=Pedestrian
# Alternatively if converted to COCO: 3=Car, 1=Person
CLASS_COLORS = {
    1: 'red',    # Pedestrian/Person (COCO or KITTI model converted)
    2: 'red',    # KITTI Pedestrian
    3: 'blue',   # Car (COCO)
}

CLASS_NAMES = {
    1: 'Person',
    2: 'Person', 
    3: 'Car'
}

def map_label(label):
    if label == 1:
        return 3 # Kitit Car -> COCO Car
    elif label == 2:
        return 1 # Kitti Pedestrian -> COCO Person
    return label # Fallback

def draw_boxes(ax, image, boxes, labels, scores=None, title="", is_mapped=False):
    if isinstance(image, torch.Tensor):
        # We need numpy arrays for plotting
        image = image.permute(1, 2, 0).cpu().numpy()
        
    ax.imshow(image)
    ax.set_title(title, fontsize=36, pad=15) # Increased title size significantly
    ax.axis('off')

    for i in range(len(boxes)):
        box = boxes[i]
        label = map_label(labels[i]) if not is_mapped else labels[i]
        score = scores[i] if scores is not None else None
        
        if label not in CLASS_NAMES:
            continue
            
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        
        color = CLASS_COLORS.get(label, 'green')
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        class_name = CLASS_NAMES.get(label, f"Class {label}")
        text = f"{class_name}"
        if score is not None:
            text += f" {score:.2f}"
            
        ax.text(xmin, max(ymin - 8, 0), text, color='white', fontsize=18, fontweight='bold', bbox=dict(facecolor=color, alpha=0.6, edgecolor='none', pad=3))

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    device = "0" if torch.cuda.is_available() else "cpu"
    
    print("Loading Models...")
    model_pretrained = UltralyticsYOLO(weights=PRETRAINED_WEIGHTS, conf=0.5, device=device, map_coco=True)
    model_finetuned = UltralyticsYOLO(weights=FINETUNED_WEIGHTS, conf=0.5, device=device, map_coco=False)
    
    print("Loading Ground Truth dataset...")
    ds = KITTIMOTS(
        root="/ghome/group01/mcv/datasets/C5/KITTI-MOTS/",
        split="validation",
        ann_source="txt",
        compute_boxes=True
    )
    
    n_frames = min(LIMIT, len(ds))
    
    # Choose samples that are far from each other
    if len(ds) > LIMIT:
        indices = np.linspace(0, len(ds) - 1, LIMIT, dtype=int)
    else:
        indices = range(len(ds))
        
    print(f"Generating visualizations for {n_frames} images...")
    for i in indices:
        img, gt_anns, meta = ds[i]
        
        # Prepare GT format
        gt_boxes = []
        gt_labels = []
        for ann in gt_anns:
            gt_boxes.append(ann.bbox_xyxy)
            gt_labels.append(ann.class_id)
            
        # Convert tensor to PIL Image for YOLO
        if isinstance(img, torch.Tensor):
            img_pil = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        else:
            img_pil = img
            
        # Run inferences directly
        pred_pre = model_pretrained.predict(img_pil)
        pred_ft = model_finetuned.predict(img_pil)
        
        if len(pred_pre["scores"]) > MAX_DET:
            idx = np.argsort(-pred_pre["scores"])[:MAX_DET]
            pred_pre["bboxes_xyxy"] = pred_pre["bboxes_xyxy"][idx]
            pred_pre["scores"] = pred_pre["scores"][idx]
            pred_pre["category_ids"] = pred_pre["category_ids"][idx]
            
        if len(pred_ft["scores"]) > MAX_DET:
            idx = np.argsort(-pred_ft["scores"])[:MAX_DET]
            pred_ft["bboxes_xyxy"] = pred_ft["bboxes_xyxy"][idx]
            pred_ft["scores"] = pred_ft["scores"][idx]
            pred_ft["category_ids"] = pred_ft["category_ids"][idx]
            
        # Manually map YOLO fine-tuned output to COCO classes.
        # YOLO training data has: 0: Car, 1: Pedestrian.
        # Expected outputs for is_mapped=True: 1 (Person), 3 (Car).
        ft_cat_ids = pred_ft["category_ids"].copy()
        ft_cat_ids_mapped = np.where(ft_cat_ids == 0, 3, ft_cat_ids) # Car -> COCO Car
        # the remaining 1s are Pedestrian -> COCO Person, already correct.
        
        # Create plot
        fig, axes = plt.subplots(3, 1, figsize=(16, 18))
        
        # GT (raw KITTI labels, needs mapping)
        draw_boxes(axes[0], img, gt_boxes, gt_labels, title="Ground Truth", is_mapped=False)
        
        # Pretrained (outputting COCO 1=Person, 3=Car natively via map_coco=True)
        draw_boxes(axes[1], img, pred_pre["bboxes_xyxy"], pred_pre["category_ids"], scores=pred_pre["scores"], title="YOLO Pre-Trained", is_mapped=True)
        
        # Fine-tuned (trained on mapped YOLO labels, mapped to COCO here)
        draw_boxes(axes[2], img, pred_ft["bboxes_xyxy"], ft_cat_ids_mapped, scores=pred_ft["scores"], title="YOLO Fine-Tuned (Full)", is_mapped=True)
        
        plt.subplots_adjust(hspace=0.05, wspace=0.0)
        
        output_file = os.path.join(RESULTS_DIR, f"comparison_{meta['seq']}_{meta['frame']:06d}.jpg")
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
    print("Qualitative evaluation completed successfully.")

if __name__ == "__main__":
    main()
