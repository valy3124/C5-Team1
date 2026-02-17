"""
COCO metrics utilities for object detection

The metrics that we include are:
- Average Precision (AP) and Average Recall (AR)
- Per-class AP/AR
- mAP across classes
- Precision-Recall and AUC curves
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def _iou_range(start: float = 0.50, stop: float = 0.95, step: float = 0.05) -> np.ndarray:
    """
    Create an IoU threshold array like COCO: 0.50, 0.55, ..., 0.95
    """
    return np.round(np.arange(start, stop + 1e-9, step), 2)

def _as_sorted_unique_ints(xs: Iterable[int]) -> List[int]:
    """
    Return sorted unique ints
    """
    return sorted(set(int(x) for x in xs))


class CocoMetrics:
    def __init__():
        # TODO
