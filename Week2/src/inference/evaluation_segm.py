from __future__ import annotations

import json
import argparse
import wandb
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils

# Import BOTH datasets
from Week2.src.datasets import KITTIMOTS, DEART

def xyxy_to_xywh(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    return [x1, y1, x2-x1, y2-y1]

class CocoSegmentationMetrics:
    def __init__(self, root: str, dataset_name: str = "kitti_mots", split: str = "validation", ann_source: str = "txt", seed: int = 42, split_ratio: float = 0.8):
        print(f"Initializing {dataset_name.upper()} dataset for SEGMENTATION from {root} (split: {split})...")
        
        # Dynamically load the correct dataset and categories
        if dataset_name == "kitti_mots":
            self.dataset = KITTIMOTS(root=root, split=split, ann_source=ann_source, compute_boxes=True, seed=seed, split_ratio=split_ratio)
            self.categories = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]
            self.label_map = self.dataset.LABELS_MAPPING # KITTI to COCO
        elif dataset_name == "deart":
            self.dataset = DEART(root=root, split=split, ann_source="xml", seed=seed, split_ratio=split_ratio)
            self.categories = [{"id": 1, "name": "person"}] # DEART only cares about humans
            self.label_map = {1: 1} # DEART Human(1) -> COCO Person(1)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.coco_gt = self._build_coco_gt()

    def _build_coco_gt(self) -> COCO:
        print("Building COCO Ground Truth dictionary for Segmentation...")
        images, annotations = [], []
        ann_id = 1
        
        for i in range(len(self.dataset)):
            img, anns, meta = self.dataset[i]
            image_id = meta["index"]
            w, h = img.size
            
            images.append({
                "id": image_id, "file_name": meta.get("image_path", ""),
                "width": w, "height": h
            })
            
            for ann in anns:
                # Ensure the class is something we want to evaluate
                if ann.class_id not in self.label_map:
                    continue
                    
                bbox = xyxy_to_xywh(ann.bbox_xyxy)
                
                # Use RLE from annotation if exists, otherwise compute it
                rle = ann.mask_rle
                    
                area = maskUtils.area(rle)

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": self.label_map[ann.class_id],
                    "segmentation": rle,
                    "bbox": bbox,
                    "area": float(area),
                    "iscrowd": 0
                })
                ann_id += 1
                
        coco = COCO()
        coco.dataset = {"images": images, "annotations": annotations, "categories": self.categories}
        coco.createIndex()
        print(f"GT Loaded: {len(annotations)} annotations across {len(images)} images.")
        return coco

    def compute_metrics(self, coco_dt: COCO) -> Dict[str, float]:
        print("Running COCO Segmentation Evaluation...")
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'segm') # Use 'segm' instead of 'bbox'
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        metrics = {
            "overall/AP_segm": stats[0], "overall/AP_50_segm": stats[1], "overall/AP_75_segm": stats[2],
            "overall/AP_small_segm": stats[3], "overall/AP_medium_segm": stats[4], "overall/AP_large_segm": stats[5],
        }
        
        print("\n--- Per-Class Segmentation Metrics ---")
        for cat_dict in self.categories:
            cat_id, cat_name = cat_dict["id"], cat_dict["name"]
            
            coco_eval_cat = COCOeval(self.coco_gt, coco_dt, 'segm')
            coco_eval_cat.params.catIds = [cat_id] 
            coco_eval_cat.evaluate()
            coco_eval_cat.accumulate()
            coco_eval_cat.summarize()
            
            cat_stats = coco_eval_cat.stats
            cat_metrics = {
                f"{cat_name}/AP_segm": cat_stats[0], f"{cat_name}/AP_50_segm": cat_stats[1], f"{cat_name}/AP_75_segm": cat_stats[2],
            }
            metrics.update(cat_metrics)
            
        return metrics
