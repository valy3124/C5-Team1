"""
Model-agnostic inference runner (Supports Faster-RCNN, DeTR & YOLO).

This script runs inference over KITTI-MOTS frames using any model wrapper.

Inference result is stored per frame:
[{
    "image_id":     int,
    "category_ids": np.ndarray (N,)   int64,
    "bboxes_xyxy":  np.ndarray (N, 4) float32,
    "scores":       np.ndarray (N,)   float32,
}]

Outputs are saved as JSONL.
"""
from __future__ import annotations

import argparse
import json
import torch
from pathlib import Path
from typing import Any, Dict
import numpy as np
from PIL import Image

from src.datasets import KITTIMOTS
from src.models.yolo import UltralyticsYOLO
from src.models.detr import HuggingFaceDETR 

def _to_jsonable_pred(image_id: int, pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a prediction dict containing NumPy arrays into JSON-serializable types.
    """
    boxes = pred.get("bboxes_xyxy")
    scores = pred.get("scores")
    category_ids = pred.get("category_ids")
    if not isinstance(boxes, np.ndarray) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("pred['bboxes_xyxy'] must be a NumPy array of shape (N,4).")
    if not isinstance(scores, np.ndarray) or scores.ndim != 1:
        raise ValueError("pred['scores'] must be a 1D NumPy array.")
    if not isinstance(category_ids, np.ndarray) or category_ids.ndim != 1:
        raise ValueError("pred['category_ids'] must be a 1D NumPy array.")

    return {
        "image_id": image_id,
        "boxes_xyxy": boxes.tolist(),
        "scores": scores.tolist(),
        "category_ids": category_ids.tolist()
    }

def build_model(args: argparse.Namespace) -> Any:
    name = args.model.lower()

    if name == "yolo":
        return UltralyticsYOLO(
            weights=args.weights,
            task="detect",
            conf=args.conf,
            device=args.device,
            half=args.half,
            max_det=args.max_det,
        )
    
    elif name == "detr":
        return HuggingFaceDETR(
            weights=args.weights,
            conf=args.conf,
            device=args.device,
            half=args.half,
            max_det=args.max_det
        )
    
    elif name == "faster_rcnn":
        # TODO
        raise NotImplementedError(f"Model {args.model} is not yet implemented.")

    raise ValueError(f"Unknown model: {args.model}")

def run_inference(
    root: str,
    split: str,
    ann_source: str,
    output: str,
    model: Any,
    limit: Optional[int] = None,
) -> None:
    """
    Run inference over KITTI-MOTS and save predictions to JSONL.

    Parameters
    ----
    root : str
        KITTI-MOTS dataset root.
    split : str
        Dataset split ("training" or "testing").
    ann_source : str
        Annotation source ("png" or "txt"). Not used for testing split.
    output : str
        Output JSONL file path.
    model : Any
        Any model object exposing predict(PIL.Image)->dict.
    limit : Optional[int]
        If provided, process only the first `limit` frames.
    """
    ds = KITTIMOTS(root=root, split=split, ann_source=ann_source, compute_boxes=True)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(ds) if limit is None else min(limit, len(ds))

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            img, _, _ = ds[i]
            pred = model.predict(img)
            f.write(json.dumps(_to_jsonable_pred(i, pred)) + "\n")

            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{n} frames")

    print(f"Saved predictions to {out_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on KITTI-MOTS dataset")

    # Inference arguments
    parser.add_argument("--root", type=str, default="~/mcv/datasets/C5/KITTI-MOTS/", help="Path to KITTI-MOTS dataset")
    parser.add_argument("--split", type=str, default="training", help="training or testing")
    parser.add_argument("--ann_source", type=str, default="txt", help="Annotation source (txt/png)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=300, help="Limit number of frames for debugging")

    # Model arguments
    parser.add_argument("--model", type=str, required=True, help="Model type: faster_rcnn, detr, yolo")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights (default: pre-trained weights of COCO)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half", action="store_true", help="Use FP16")
    parser.add_argument("--max_det", type=int, default=100, help="Limit number of detections per image (COCO metrics only use top 100)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = build_model(args)

    run_inference(
        root=args.root,
        split=args.split,
        ann_source=args.ann_source,
        output=args.output,
        model=model,
        limit=args.limit,
    )