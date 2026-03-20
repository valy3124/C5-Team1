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
from datasets import KITTIMOTS, DEART

def xyxy_to_xywh(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    return [x1, y1, x2 - x1, y2 - y1]


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
                f"{cat_name}/AP_segm":    cat_stats[0],
                f"{cat_name}/AP_50_segm": cat_stats[1],
                f"{cat_name}/AP_75_segm": cat_stats[2],
            }

            for t_idx, iou_t in enumerate(iou_thresholds):
                # Mean precision across all recall points at this IoU threshold
                prec_t  = coco_eval_cat.eval["precision"][t_idx, :, 0, 0, 2]
                valid_p = prec_t[prec_t > -1]
                p_t     = float(np.mean(valid_p)) if len(valid_p) > 0 else 0.0

                # Recall at this IoU threshold (area=all, maxDets=100)
                rec_raw = coco_eval_cat.eval["recall"][t_idx, 0, 0, 2]
                rec_t   = float(rec_raw) if np.ndim(rec_raw) == 0 else float(np.mean(rec_raw))

                f1_t   = 2 * p_t * rec_t / (p_t + rec_t + 1e-6)
                suffix = str(round(iou_t * 100))
                cat_metrics[f"{cat_name}/F1_score_{suffix}_segm"] = f1_t
                cat_metrics[f"{cat_name}/Recall_{suffix}_segm"]   = rec_t

                all_class_f1s[iou_t].append(f1_t)
                all_class_recalls[iou_t].append(rec_t)

            metrics.update(cat_metrics)
            
        # Attach the global coco_eval for optional error analysis
        self._last_coco_eval = coco_eval

        # Overall (macro-averaged across categories) F1 and Recall
        for iou_t in iou_thresholds:
            suffix = str(round(iou_t * 100))
            metrics[f"overall/F1_score_{suffix}_segm"] = (
                float(np.mean(all_class_f1s[iou_t])) if all_class_f1s[iou_t] else 0.0
            )
            metrics[f"overall/Recall_{suffix}_segm"] = (
                float(np.mean(all_class_recalls[iou_t])) if all_class_recalls[iou_t] else 0.0
            )

        # Mean F1 across all thresholds (analogous to mAP)
        metrics["overall/mF1_segm"] = float(np.mean(
            [metrics[f"overall/F1_score_{round(t * 100)}_segm"] for t in iou_thresholds]
        ))

        return metrics

    def compute_error_analysis(self, coco_dt: COCO) -> Dict[str, float]:
        """
        Compute FP / Recall error analysis after evaluation.

        Uses the COCOeval already run by compute_metrics (stored as
        self._last_coco_eval) if available, otherwise re-runs it.

        Returns
        -------
        dict with keys:
          error_analysis/total_fp          – number of FP detections at IoU≥0.50
          error_analysis/avg_fp_confidence – mean score of those FPs
          error_analysis/recall_at_50      – overall recall at IoU=0.50
          error_analysis/total_tp          – number of TP detections at IoU≥0.50
          error_analysis/total_gt          – total GT annotations
        """
        coco_eval = getattr(self, "_last_coco_eval", None)
        if coco_eval is None or not hasattr(coco_eval, "evalImgs"):
            print("Re-running COCOeval for error analysis...")
            coco_eval = COCOeval(self.coco_gt, coco_dt, "segm")
            coco_eval.evaluate()
            coco_eval.accumulate()

        # ── Extract per-image TP / FP from evalImgs ──────────────────────
        # evalImgs is a list (one entry per image × category × area-range × maxDet).
        # We use iouThr index 0 → IoU = 0.50 (matches COCO AP@0.50 threshold).
        iou_idx = 0   # corresponds to iouThrs[0] = 0.50

        total_tp   = 0
        total_fp   = 0
        fp_scores  = []
        total_gt   = 0
        per_detection_tp: dict = {}   # det_id (int) → 1=TP, 0=FP

        for ev in coco_eval.evalImgs:
            if ev is None:
                continue
            dt_ids     = ev["dtIds"]                  # list of detection IDs
            dt_matches = np.array(ev["dtMatches"])    # shape (n_iou, n_dt)
            dt_ignore  = np.array(ev["dtIgnore"])     # shape (n_iou, n_dt)
            dt_scores  = np.array(ev["dtScores"])     # shape (n_dt,)
            gt_ignore  = np.array(ev["gtIgnore"])    # shape (n_gt,)

            n_dt = dt_scores.shape[0]
            total_gt += int((gt_ignore == 0).sum())

            for d in range(n_dt):
                det_id = dt_ids[d]
                if dt_ignore[iou_idx, d]:
                    # Ignored detection — mark as -1 so CSV shows it clearly
                    per_detection_tp[det_id] = -1
                    continue
                if dt_matches[iou_idx, d] > 0:
                    total_tp += 1
                    per_detection_tp[det_id] = 1
                else:
                    total_fp += 1
                    per_detection_tp[det_id] = 0
                    fp_scores.append(float(dt_scores[d]))

        avg_fp_conf = float(np.mean(fp_scores)) if fp_scores else 0.0
        recall_at_50 = total_tp / total_gt if total_gt > 0 else 0.0

        print(f"\n--- Error Analysis (IoU ≥ 0.50) ---")
        print(f"  Total GT annotations : {total_gt}")
        print(f"  Total TP             : {total_tp}")
        print(f"  Total FP             : {total_fp}")
        print(f"  Recall@0.50          : {recall_at_50:.4f}")
        print(f"  Avg FP confidence    : {avg_fp_conf:.4f}")

        summary = {
            "error_analysis/total_fp":          total_fp,
            "error_analysis/total_tp":          total_tp,
            "error_analysis/total_gt":          total_gt,
            "error_analysis/avg_fp_confidence": avg_fp_conf,
            "error_analysis/recall_at_50":      recall_at_50,
        }
        return summary, per_detection_tp
