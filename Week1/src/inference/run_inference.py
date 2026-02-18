"""
Model-agnostic inference runner.

This script runs inference over KITTI-MOTS frames using any model wrapper

The output is a dict with:
    - "boxes_xyxy": np.ndarray (N, 4) float32
    - "scores":     np.ndarray (N,)   float32
    - "classes":    np.ndarray (N,)   int64
    - "masks":      None
    - Plus metadata (sequence, frame...)

Outputs are saved as JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from src.datasets import KITTIMOTS
from src.models.yolo import UltralyticsYOLO


def _to_jsonable_pred(pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a prediction dict containing NumPy arrays into JSON-serializable types.
    """
    boxes = pred.get("boxes_xyxy")
    scores = pred.get("scores")
    classes = pred.get("classes")
    masks = pred.get("masks", None)

    if not isinstance(boxes, np.ndarray) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("pred['boxes_xyxy'] must be a NumPy array of shape (N,4).")
    if not isinstance(scores, np.ndarray) or scores.ndim != 1:
        raise ValueError("pred['scores'] must be a 1D NumPy array.")
    if not isinstance(classes, np.ndarray) or classes.ndim != 1:
        raise ValueError("pred['classes'] must be a 1D NumPy array.")
    if masks is not None and (not isinstance(masks, np.ndarray) or masks.ndim != 3):
        raise ValueError("pred['masks'] must be None or a NumPy array of shape (N,H,W).")

    return {
        "boxes_xyxy": boxes.tolist(),
        "scores": scores.tolist(),
        "classes": classes.tolist(),
        "masks": None,
    }


def build_model(args: argparse.Namespace) -> Any:
    """
    Build the requested model wrapper.
    """
    #TODO: Now only yolo!!!!
    name = args.model.lower()

    if name in {"yolo"}:
        return UltralyticsYOLO(
            weights=args.weights,
            task="detect",
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            max_det=args.max_det,
        )

    raise ValueError(f"Unknown model: {args.model}")


def run_inference(
    *,
    root: str,
    split: str,
    ann_source: str,
    output_path: str,
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
    output_path : str
        Output JSONL file path.
    model : Any
        Any model object exposing predict(PIL.Image)->dict.
    limit : Optional[int]
        If provided, process only the first `limit` frames.
    """
    ds = KITTIMOTS(root=root, split=split, ann_source=ann_source, compute_boxes=True)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(ds) if limit is None else min(limit, len(ds))

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            img, _anns, meta = ds[i]
            pred = model.predict(img)
            pred_json = _to_jsonable_pred(pred)

            record = {
                "index": int(meta["index"]),
                "seq": meta["seq"],
                "frame": int(meta["frame"]),
                "image_path": meta["image_path"],
                "image_size": [int(img.size[0]), int(img.size[1])],  # [W, H]
                "pred": pred_json,
            }

            f.write(json.dumps(record) + "\n")

            if (i + 1) % 100 == 0 or (i + 1) == n:
                print(f"[run_inference] Processed {i+1}/{n} frames")

    print(f"[run_inference] Saved predictions to: {out_path.resolve()}")

  #TODO
def parse_args():
    pass

def main() -> None:
    args = parse_args()
    model = build_model(args)

    run_inference(
        root=args.root,
        split=args.split,
        ann_source=args.ann_source,
        output_path=args.output,
        model=model,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
