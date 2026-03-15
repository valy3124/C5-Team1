from __future__ import annotations

import numpy as np
from typing import Dict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as maskUtils

from Week2.src.datasets import KITTIMOTS, DEART


def xyxy_to_xywh(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    return [x1, y1, x2 - x1, y2 - y1]


class CocoSegmentationMetrics:
    """
    Builds a COCO-format ground-truth index and computes segmentation metrics,
    including AP (overall, @50, @75, by size) and F1/Recall at each
    COCOeval IoU threshold (0.50 : 0.05 : 0.95) for every category.
    """

    def __init__(
        self,
        root: str,
        dataset_name: str = "kitti_mots",
        split: str = "validation",
        ann_source: str = "txt",
        seed: int = 42,
        split_ratio: float = 0.8,
    ):
        print(f"Initializing {dataset_name.upper()} dataset for SEGMENTATION "
              f"from {root} (split: {split})...")

        if dataset_name == "kitti_mots":
            self.dataset = KITTIMOTS(
                root=root, split=split, ann_source=ann_source,
                compute_boxes=True, seed=seed, split_ratio=split_ratio,
            )
            self.categories = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]
            self.label_map = self.dataset.LABELS_MAPPING  # KITTI → COCO class IDs
        elif dataset_name == "deart":
            self.dataset = DEART(
                root=root, split=split, ann_source="xml",
                seed=seed, split_ratio=split_ratio,
            )
            self.categories = [{"id": 1, "name": "person"}]
            self.label_map = {1: 1}  # DEART Human → COCO Person
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.coco_gt = self._build_coco_gt()

    def _build_coco_gt(self) -> COCO:
        print("Building COCO Ground Truth index for Segmentation...")
        images, annotations = [], []
        ann_id = 1

        for i in range(len(self.dataset)):
            img, anns, meta = self.dataset[i]
            image_id = meta["index"]
            w, h = img.size

            images.append({
                "id": image_id,
                "file_name": meta.get("image_path", ""),
                "width": w,
                "height": h,
            })

            for ann in anns:
                if ann.class_id not in self.label_map:
                    continue

                rle = ann.mask_rle
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": self.label_map[ann.class_id],
                    "segmentation": rle,
                    "bbox": xyxy_to_xywh(ann.bbox_xyxy),
                    "area": float(maskUtils.area(rle)),
                    "iscrowd": 0,
                })
                ann_id += 1

        coco = COCO()
        coco.dataset = {"images": images, "annotations": annotations, "categories": self.categories}
        coco.createIndex()
        print(f"GT loaded: {len(annotations)} annotations across {len(images)} images.")
        return coco

    def compute_metrics(self, coco_dt: COCO) -> Dict[str, float]:
        """
        Runs COCO segmentation evaluation and returns a metrics dict containing:
          - overall/AP_segm, AP_50_segm, AP_75_segm, AP_{small,medium,large}_segm
          - per-category AP_segm, AP_50_segm, AP_75_segm
          - per-category and overall F1_score_{T}_segm / Recall_{T}_segm
            for each IoU threshold T in [50, 55, …, 95]
          - overall/mF1_segm (mean F1 across all thresholds)
        """
        print("Running COCO Segmentation Evaluation...")
        coco_eval = COCOeval(self.coco_gt, coco_dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        metrics: Dict[str, float] = {
            "overall/AP_segm":       stats[0],
            "overall/AP_50_segm":    stats[1],
            "overall/AP_75_segm":    stats[2],
            "overall/AP_small_segm": stats[3],
            "overall/AP_medium_segm":stats[4],
            "overall/AP_large_segm": stats[5],
        }

        # F1 and Recall at each COCOeval IoU threshold (T ∈ {0.50, 0.55, …, 0.95}).
        # Precision tensor shape: [T, R, K, A, M]
        #   T = IoU thresholds, R = recall points, K = categories, A = area ranges, M = max-dets
        iou_thresholds = coco_eval.params.iouThrs
        all_class_f1s    = {t: [] for t in iou_thresholds}
        all_class_recalls = {t: [] for t in iou_thresholds}

        print("\n--- Per-Class Segmentation Metrics ---")
        for cat_dict in self.categories:
            cat_id, cat_name = cat_dict["id"], cat_dict["name"]

            coco_eval_cat = COCOeval(self.coco_gt, coco_dt, "segm")
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
