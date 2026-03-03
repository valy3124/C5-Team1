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
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# Set paths
RESULTS_DIR = os.path.join(BASE_DIR, "results/rt_detr/qualitative_evaluation")

# Models info
FINETUNED_WEIGHTS = os.path.join(BASE_DIR, "results/rt_detr/BEST_rt_detr_base_Frz1_LR1.0e-05_Opt_adamw_Sch_cosine_GC_0.8_acmnqklx/best_model.pth")
PRETRAINED_WEIGHTS = "PekingU/rtdetr_r50vd"

# Limits for evaluation
LIMIT = 100

# Dataset class mapping
# We want everything mapped to standard IDs for display:
# 1 = Person
# 3 = Car

CLASS_COLORS = {
    1: 'red',    # Person
    3: 'blue',   # Car
}

CLASS_NAMES = {
    1: 'Person',
    3: 'Car'
}

def map_gt_label(label):
    # GT comes from KITTIMOTS: 1 = Car, 2 = Pedestrian
    if label == 1:
        return 3 # Car
    elif label == 2:
        return 1 # Person
    return label 

def draw_boxes(ax, image, boxes, labels, scores=None, title="", mapping_func=None):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        
    ax.imshow(image)
    ax.set_title(title, fontsize=36, pad=15)
    ax.axis('off')

    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        if mapping_func:
            label = mapping_func(label)

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

class HuggingFaceRTDETR:
    def __init__(self, weights=None, conf=0.5, device="0", is_pretrained=False):
        self.conf = conf
        self.device = device
        hf_source = "PekingU/rtdetr_r50vd"
        
        self.processor = RTDetrImageProcessor.from_pretrained(hf_source)
        
        if is_pretrained:
            self.model = RTDetrForObjectDetection.from_pretrained(hf_source)
        else:
            self.model = RTDetrForObjectDetection.from_pretrained(
                hf_source, 
                ignore_mismatched_sizes=True, 
                num_labels=3 # Trained with 3 classes (Background, Car, Pedestrian)
            )
            state_dict = torch.load(weights, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
                
            cls_weight_key = "class_labels_classifier.weight"
            if cls_weight_key in state_dict:
                num_classes_in_ckpt = state_dict[cls_weight_key].shape[0]
                if self.model.config.num_labels != num_classes_in_ckpt:
                    self.model.config.num_labels = num_classes_in_ckpt
                    d_model = self.model.config.d_model
                    self.model.class_labels_classifier = torch.nn.Linear(d_model, num_classes_in_ckpt)
            
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        if not isinstance(image, list):
            image = [image]

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([img.size[::-1] for img in image]).to(self.device)

        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.conf,
        )

        outputs_list = []
        for result in results:
            outputs_list.append({
                "bboxes_xyxy": result["boxes"].detach().cpu().numpy().astype(np.float32),
                "scores": result["scores"].detach().cpu().numpy().astype(np.float32),
                "category_ids": result["labels"].detach().cpu().numpy().astype(np.int64),
            })

        return outputs_list[0]

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Models...")
    model_finetuned = HuggingFaceRTDETR(weights=FINETUNED_WEIGHTS, conf=0.25, device=device, is_pretrained=False)
    
    print("Loading Ground Truth dataset...")
    ds = KITTIMOTS(
        root="/ghome/group01/mcv/datasets/C5/KITTI-MOTS/",
        split="validation",
        ann_source="txt",
        compute_boxes=True
    )
    
    n_frames = min(LIMIT, len(ds))
    
    if len(ds) > LIMIT:
        indices = np.linspace(0, len(ds) - 1, LIMIT, dtype=int)
    else:
        indices = range(len(ds))
        
    print(f"Generating visualizations for {n_frames} images...")
        
    # Finetuned mapping: fine_tune_rt_detr trained uses model labels: 1=Car, 2=Pedestrian
    def map_finetuned(label):
        if label == 1: return 3 # Car
        if label == 2: return 1 # Person
        return -1

    for i in indices:
        img, gt_anns, meta = ds[i]
        
        gt_boxes = []
        gt_labels = []
        for ann in gt_anns:
            gt_boxes.append(ann.bbox_xyxy)
            gt_labels.append(ann.class_id)
            
        if isinstance(img, torch.Tensor):
            img_pil = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        else:
            img_pil = img
            
        pred_ft = model_finetuned.predict(img_pil)
        
        # Apply strict thresholds: 0.5 for Car (1), 0.25 for Pedestrian (2)
        keep = []
        for j in range(len(pred_ft["scores"])):
            cat = pred_ft["category_ids"][j]
            score = pred_ft["scores"][j]
            if cat == 1 and score < 0.5:
                continue
            keep.append(j)
            
        pred_ft["bboxes_xyxy"] = pred_ft["bboxes_xyxy"][keep]
        pred_ft["scores"] = pred_ft["scores"][keep]
        pred_ft["category_ids"] = pred_ft["category_ids"][keep]
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        draw_boxes(axes[0], img, gt_boxes, gt_labels, title="Ground Truth", mapping_func=map_gt_label)
        draw_boxes(axes[1], img, pred_ft["bboxes_xyxy"], pred_ft["category_ids"], scores=pred_ft["scores"], title="RT-DETR Fine-Tuned (Full)", mapping_func=map_finetuned)
        
        plt.subplots_adjust(hspace=0.05, wspace=0.0)
        
        output_file = os.path.join(RESULTS_DIR, f"comparison_{meta['seq']}_{meta['frame']:06d}.jpg")
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
    print("Qualitative evaluation completed successfully.")

if __name__ == "__main__":
    main()
