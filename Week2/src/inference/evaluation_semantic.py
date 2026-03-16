"""
evaluation_semantic.py

Utilities for semantic segmentation evaluation on KITTI-MOTS.

The semantic label space used throughout this file:
    0 = background
    1 = person  (COCO id 1)
    3 = car     (COCO id 3)

Metrics computed
----------------
* Per-class IoU  = |pred_c ∩ gt_c| / |pred_c ∪ gt_c|
* mIoU           = mean per-class IoU (background excluded by default)
* Per-class pixel accuracy
* Mean pixel accuracy
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

CLASS_IDS   = [1, 3]          # person, car  (background=0 excluded from mIoU)
CLASS_NAMES = {1: "person", 3: "car", 0: "background"}
ALL_CLASS_IDS = [0, 1, 3]     # including background for pixel-acc


# ---------------------------------------------------------------------------
# Per-image helpers
# ---------------------------------------------------------------------------

def compute_iou_for_class(
    pred: np.ndarray,
    gt: np.ndarray,
    class_id: int,
) -> float:
    """IoU for a single class on a single image.  Returns NaN if class absent."""
    pred_c = pred == class_id
    gt_c   = gt   == class_id
    intersection = np.logical_and(pred_c, gt_c).sum()
    union        = np.logical_or(pred_c,  gt_c).sum()
    if union == 0:
        return float("nan")   # class not present → excluded from mean
    return float(intersection) / float(union)


def compute_pixel_acc_for_class(
    pred: np.ndarray,
    gt: np.ndarray,
    class_id: int,
) -> float:
    """Pixel accuracy for *class_id*: correct / total gt pixels for that class."""
    gt_c   = gt == class_id
    n_gt   = gt_c.sum()
    if n_gt == 0:
        return float("nan")
    correct = np.logical_and(pred == class_id, gt_c).sum()
    return float(correct) / float(n_gt)


def compute_miou_single(
    pred: np.ndarray,
    gt: np.ndarray,
    class_ids: List[int] = CLASS_IDS,
) -> Tuple[float, Dict[int, float]]:
    """
    Compute mIoU and per-class IoU for a single (pred, gt) pair.

    Returns
    -------
    miou : float
        Mean IoU over *class_ids* (NaN classes skipped).
    per_class : dict[int, float]
        IoU keyed by class id.  NaN if class absent in both maps.
    """
    per_class = {c: compute_iou_for_class(pred, gt, c) for c in class_ids}
    valid = [v for v in per_class.values() if not np.isnan(v)]
    miou  = float(np.mean(valid)) if valid else float("nan")
    return miou, per_class


# ---------------------------------------------------------------------------
# Accumulator across the whole dataset
# ---------------------------------------------------------------------------

class SemanticEvaluator:
    """
    Accumulates per-image semantic segmentation results and produces
    dataset-level metrics.

    Usage
    -----
    >>> evaluator = SemanticEvaluator()
    >>> for pred_map, gt_map in ...:
    ...     evaluator.update(pred_map, gt_map)
    >>> metrics = evaluator.compute()
    """

    def __init__(self, class_ids: List[int] = CLASS_IDS):
        self.class_ids = class_ids
        # Confusion-matrix approach: accumulate intersection & union per class
        self._intersection: Dict[int, float] = {c: 0.0 for c in class_ids}
        self._union:        Dict[int, float] = {c: 0.0 for c in class_ids}
        self._gt_count:     Dict[int, float] = {c: 0.0 for c in class_ids}
        self._pred_count:   Dict[int, float] = {c: 0.0 for c in class_ids}
        self._n_images = 0
        # Also track per-image mIoU for variance estimation
        self._per_image_miou: List[float] = []

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """Register one (pred, gt) semantic map pair."""
        self._n_images += 1
        image_ious = []
        for c in self.class_ids:
            pred_c = pred == c
            gt_c   = gt   == c
            inter  = float(np.logical_and(pred_c, gt_c).sum())
            union  = float(np.logical_or(pred_c,  gt_c).sum())
            self._intersection[c] += inter
            self._union[c]        += union
            self._gt_count[c]     += float(gt_c.sum())
            self._pred_count[c]   += float(pred_c.sum())
            if union > 0:
                image_ious.append(inter / union)
        if image_ious:
            self._per_image_miou.append(float(np.mean(image_ious)))

    def compute(self) -> Dict[str, float]:
        """
        Return a dict with:
          - ``overall/mIoU``
          - ``overall/mean_pixel_acc``
          - ``<class_name>/IoU``     for each class
          - ``<class_name>/pixel_acc`` for each class
          - ``n_images``
        """
        metrics: Dict[str, float] = {}
        ious, pixel_accs = [], []

        for c in self.class_ids:
            name  = CLASS_NAMES.get(c, str(c))
            union = self._union[c]
            iou   = self._intersection[c] / union if union > 0 else float("nan")
            metrics[f"{name}/IoU"] = iou
            if not np.isnan(iou):
                ious.append(iou)

            gt_total = self._gt_count[c]
            pacc = self._intersection[c] / gt_total if gt_total > 0 else float("nan")
            metrics[f"{name}/pixel_acc"] = pacc
            if not np.isnan(pacc):
                pixel_accs.append(pacc)

        metrics["overall/mIoU"]           = float(np.mean(ious))       if ious       else float("nan")
        metrics["overall/mean_pixel_acc"] = float(np.mean(pixel_accs)) if pixel_accs else float("nan")
        metrics["n_images"] = float(self._n_images)

        # Per-image mIoU statistics
        if self._per_image_miou:
            metrics["overall/mIoU_std"]    = float(np.std(self._per_image_miou))
            metrics["overall/mIoU_median"] = float(np.median(self._per_image_miou))

        return metrics

    def reset(self) -> None:
        self.__init__(self.class_ids)
