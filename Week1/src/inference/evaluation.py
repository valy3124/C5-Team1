"""
COCO metrics evaluation script for KITTI-MOTS and DEART (Domain Shift).
"""
from __future__ import annotations

import json
import argparse
import wandb
from pathlib import Path
from typing import Dict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Import BOTH datasets
from src.datasets import KITTIMOTS, DEART

def xyxy_to_xywh(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    return [x1, y1, x2-x1, y2-y1]

class CocoMetrics:
    def __init__(self, root: str, dataset_name: str = "kitti_mots", split: str = "validation", ann_source: str = "txt", seed: int = 42, split_ratio: float = 0.8):
        print(f"Initializing {dataset_name.upper()} dataset from {root} (split: {split})...")
        
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
        print("Building COCO Ground Truth dictionary...")
        images, annotations = [], []
        ann_id = 1
        
        for i in range(len(self.dataset)):
            img, anns, meta = self.dataset[i]
            image_id = meta["index"]
            w, h = img.size
            
            images.append({
                "id": image_id, "file_name": meta["image_path"],
                "width": w, "height": h
            })
            
            for ann in anns:
                # Ensure the class is something we want to evaluate
                if ann.class_id not in self.label_map:
                    continue
                    
                bbox = xyxy_to_xywh(ann.bbox_xyxy)
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": self.label_map[ann.class_id],
                    "bbox": bbox,
                    "area": bbox[2]*bbox[3],
                    "iscrowd": 0
                })
                ann_id += 1
                
            if (i+1) % 100 == 0:
                print(f"Processed GT for {i+1}/{len(self.dataset)} frames...")

        coco = COCO()
        coco.dataset = {"images": images, "annotations": annotations, "categories": self.categories}
        coco.createIndex()
        print(f"GT Loaded: {len(annotations)} annotations across {len(images)} images.")
        return coco

    def compute_metrics(self, coco_dt: COCO) -> Dict[str, float]:
        print("Running COCO Evaluation...")
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        metrics = {
            "overall/AP": stats[0], "overall/AP_50": stats[1], "overall/AP_75": stats[2],
            "overall/AP_small": stats[3], "overall/AP_medium": stats[4], "overall/AP_large": stats[5],
        }
        
        print("\n--- Per-Class Metrics ---")
        for cat_dict in self.categories:
            cat_id, cat_name = cat_dict["id"], cat_dict["name"]
            
            coco_eval_cat = COCOeval(self.coco_gt, coco_dt, 'bbox')
            coco_eval_cat.params.catIds = [cat_id] 
            coco_eval_cat.evaluate()
            coco_eval_cat.accumulate()
            coco_eval_cat.summarize()
            
            cat_stats = coco_eval_cat.stats
            cat_metrics = {
                f"{cat_name}/AP": cat_stats[0], f"{cat_name}/AP_50": cat_stats[1], f"{cat_name}/AP_75": cat_stats[2],
            }
            metrics.update(cat_metrics)
            
        return metrics

    def evaluate(self, pred_path: str):
        print(f"Loading predictions from {pred_path}...")
        coco_results = []
        
        with open(pred_path, 'r') as f:
            for line in f:
                r = json.loads(line)
                image_id = r["image_id"]
                for bbox, score, cat_id in zip(r["bboxes_xyxy"], r["scores"], r["category_ids"]):
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": xyxy_to_xywh(bbox),
                        "score": score
                    })

        if not coco_results:
            raise ValueError("No predictions found in the provided JSONL file!")
        
        coco_dt = self.coco_gt.loadRes(coco_results)
        metrics = self.compute_metrics(coco_dt)
        if wandb.run is not None:
            wandb.log(metrics)
        return metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on datasets")
    parser.add_argument("--dataset", type=str, default="kitti_mots", choices=["kitti_mots", "deart"], help="Dataset to evaluate on")
    parser.add_argument("--root", type=str, default="~/mcv/datasets/C5/KITTI-MOTS/", help="Path to dataset")
    parser.add_argument("--split", type=str, default="validation", help="Split to evaluate on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Split ratio for training")
    parser.add_argument("--preds", required=True, help="Path to predictions.jsonl")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    metrics = CocoMetrics(root=args.root, dataset_name=args.dataset, split=args.split, seed=args.seed, split_ratio=args.split_ratio)

    filename = Path(args.preds).name
    exp_name = filename.removesuffix("_results.jsonl")
    if exp_name == filename: exp_name = "default_experiment"
    
    wandb.init(entity="c5-team1", project="COCO-evaluation", name=exp_name)
    metrics.evaluate(pred_path=args.preds)
    wandb.finish()