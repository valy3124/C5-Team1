"""
COCO metrics evaluation script for KITTI-MOTS.

We include the metrics from https://cocodataset.org/#detection-eval (overall and per-class).
"""
from __future__ import annotations

import json
import argparse
import wandb
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.datasets import KITTIMOTS

def xyxy_to_xywh(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    return [x1, y1, x2-x1, y2-y1]

class CocoMetrics:
    def __init__(self, root: str, split: str = "validation", ann_source: str = "txt"):
        print(f"Initializing KITTI-MOTS dataset from {root} (split: {split})...")
        self.dataset = KITTIMOTS(root=root, split=split, ann_source=ann_source, compute_boxes=True)
        self.categories = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]
        
        # Build COCO GT object immediately
        self.coco_gt = self._build_coco_gt()

    def _build_coco_gt(self) -> COCO:
        print("Building COCO Ground Truth dictionary...")

        images = [] # COCO format for images
        annotations = [] # COCO format for annotations
        categories = self.categories # COCO format for KITTI-MOTS categories
        
        # Iterate over the dataset to populate GT
        ann_id = 1
        for i in range(len(self.dataset)):
            img, anns, meta = self.dataset[i]

            image_id = meta["index"]
            w, h = img.size
            images.append({
                "id": image_id,
                "file_name": meta["image_path"],
                "width": w,
                "height": h
            })
            
            for ann in anns:
                bbox = xyxy_to_xywh(ann.bbox_xyxy)
                
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": self.dataset.LABELS_MAPPING[ann.class_id],
                    "bbox": bbox,
                    "area": bbox[2]*bbox[3],
                    "iscrowd": 0
                })
                ann_id += 1
                
            if (i+1) % 100 == 0:
                print(f"Processed GT for {i+1}/{len(self.dataset)} frames...")

        coco = COCO()
        coco.dataset = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        coco.createIndex()
        print(f"GT Loaded: {len(annotations)} annotations across {len(images)} images.")
        return coco

    def evaluate(self, pred_path: str):
        print(f"Loading predictions from {pred_path}...")
        coco_results = []
        
        with open(pred_path, 'r') as f:
            for line in f:
                r = json.loads(line)
                
                # Extract inference results
                image_id = r["image_id"]
                for bbox, score, cat_id in zip(r["bboxes_xyxy"], r["scores"], r["category_ids"]):
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": cat_id,  # Already mapped to COCO at inference time
                        "bbox": xyxy_to_xywh(bbox),
                        "score": score
                    })

        if not coco_results:
            raise ValueError("No predictions found in the provided JSONL file!")
        
        # Build COCO DT object from predictions
        coco_dt = self.coco_gt.loadRes(coco_results)

        print("Running COCO Evaluation...")
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        wandb.log({
            "overall/AP": coco_eval.stats[0],
            "overall/AP_50": coco_eval.stats[1],
            "overall/AP_75": coco_eval.stats[2],
            "overall/AP_small": coco_eval.stats[3],
            "overall/AP_medium": coco_eval.stats[4],
            "overall/AP_large": coco_eval.stats[5],
            "overall/AR_max1": coco_eval.stats[6],
            "overall/AR_max10": coco_eval.stats[7],
            "overall/AR_max100": coco_eval.stats[8],
            "overall/AR_small": coco_eval.stats[9],
            "overall/AR_medium": coco_eval.stats[10],
            "overall/AR_large": coco_eval.stats[11],
        })
        
        print("\n--- Per-Class Metrics ---")
        for cat_dict in self.categories:
            cat_id, cat_name = cat_dict["id"], cat_dict["name"]
            print(f"\nCategory: {cat_name.capitalize()}")

            coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
            coco_eval.params.catIds = [cat_id] # Only evaluate this category
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            wandb.log({
                f"{cat_name}/AP": coco_eval.stats[0],
                f"{cat_name}/AP_50": coco_eval.stats[1],
                f"{cat_name}/AP_75": coco_eval.stats[2],
                f"{cat_name}/AP_small": coco_eval.stats[3],
                f"{cat_name}/AP_medium": coco_eval.stats[4],
                f"{cat_name}/AP_large": coco_eval.stats[5],
                f"{cat_name}/AR_max1": coco_eval.stats[6],
                f"{cat_name}/AR_max10": coco_eval.stats[7],
                f"{cat_name}/AR_max100": coco_eval.stats[8],
                f"{cat_name}/AR_small": coco_eval.stats[9],
                f"{cat_name}/AR_medium": coco_eval.stats[10],
                f"{cat_name}/AR_large": coco_eval.stats[11],
            })