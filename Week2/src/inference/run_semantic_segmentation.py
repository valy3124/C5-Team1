"""
run_semantic_segmentation.py

Semantic segmentation on KITTI-MOTS with two complementary approaches:

  --mode text_prompt
      Zero-shot: GroundingDINO detects objects from free-form text labels,
      SAM segments them, and masks are merged per predicted class to form a
      semantic map.  *No ground-truth information used at inference.*

  --mode pretrained_sam
      Supervised (oracle boxes): SAM (pretrained weights) is prompted with
      GT bounding boxes.  Each predicted mask inherits the GT class label.
      Masks are merged per class into a semantic map.

  --mode finetuned_sam
      Same as pretrained_sam but using the finetuned SAM checkpoint
      (requires --weights).  This is the primary comparison target.

All modes evaluate against GT semantic maps derived from instance-level
KITTI-MOTS annotations (instance masks merged by COCO class id).

Semantic label space
--------------------
    0 = background
    1 = person  (COCO id)
    3 = car     (COCO id)

Outputs (under --output_dir/<exp_name>/)
-----------------------------------------
    metrics.json            — mIoU, per-class IoU, pixel accuracy
    viz/NNNN_*.png          — side-by-side visualisations
    semantic_preds.npz      — stacked predicted semantic maps (optional)

  --mode rich_text_prompt
      Open-vocabulary: segment everything visible using many text labels
      (full COCO + driving-specific classes).  No GT metrics — qualitative
      only.  Shows that the pipeline generalises far beyond person + car.

Usage (from Week2/ directory)
-------------------------------
    # Zero-shot text-prompted
    python -m src.inference.run_semantic_segmentation \\
        --mode text_prompt --text_labels "person. car."

    # Pretrained SAM with GT-box prompts
    python -m src.inference.run_semantic_segmentation --mode pretrained_sam

    # Finetuned SAM with GT-box prompts
    python -m src.inference.run_semantic_segmentation \\
        --mode finetuned_sam \\
        --weights results_finetune/sam_base_lh35g5yk/best_model.pth

    # Rich open-vocabulary (qualitative only, no GT needed)
    python -m src.inference.run_semantic_segmentation --mode rich_text_prompt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import SamModel, SamProcessor
import pycocotools.mask as rletools

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent   # Week2/
sys.path.insert(0, str(ROOT / "src"))

from datasets import KITTIMOTS
from models.grounded_sam import GroundedSamWrapper
from prompting.text import TextPromptStrategy
from inference.evaluation_semantic import SemanticEvaluator, CLASS_IDS, CLASS_NAMES


# ---------------------------------------------------------------------------
# Rich open-vocabulary label set (driving-scene oriented)
# ---------------------------------------------------------------------------

# Covers the 80 COCO classes most likely to appear in driving scenes plus
# a few extra outdoor / urban categories that GroundingDINO handles well.
RICH_LABELS = (
    "person . car . truck . bus . motorcycle . bicycle . "
    "traffic light . stop sign . fire hydrant . parking meter . "
    "bench . bird . cat . dog . horse . cow . "
    "backpack . umbrella . handbag . suitcase . "
    "bottle . cup . chair . potted plant . "
    "tree . building . wall . fence . pole . road sign . "
    "sky . road . sidewalk . vegetation . terrain ."
)

# Generate a visually distinct palette for up to 40 classes
# using a hand-picked set of well-separated hues.
_RICH_BASE_COLORS: List[Tuple[int, int, int]] = [
    (220,  50,  50),  # red
    ( 50, 100, 220),  # blue
    ( 50, 180,  50),  # green
    (255, 165,   0),  # orange
    (160,  50, 200),  # purple
    (  0, 200, 200),  # cyan
    (230, 220,  50),  # yellow
    (255, 100, 180),  # pink
    (100, 200, 100),  # light green
    (200, 100,  50),  # brown
    ( 50, 220, 180),  # teal
    (180, 180,  50),  # olive
    (220,  50, 150),  # rose
    ( 50, 150, 255),  # sky blue
    (255, 200, 100),  # peach
    (100,  50, 200),  # indigo
    (  0, 255, 128),  # mint
    (255,  80,  80),  # salmon
    (128, 255,   0),  # lime
    (200,   0, 100),  # crimson
    ( 80, 200, 255),  # light blue
    (255, 230,   0),  # gold
    (150,  80, 255),  # violet
    (  0, 180,  80),  # emerald
    (255, 140,  60),  # amber
    (100, 255, 200),  # aquamarine
    (220,  80, 220),  # magenta
    ( 60, 120,  60),  # forest green
    (200, 200, 255),  # lavender
    (255,  50, 200),  # hot pink
    (160, 100,  50),  # tan
    ( 50,  50, 200),  # navy
    (200, 255, 100),  # yellow-green
    (255, 100,   0),  # deep orange
    (  0, 100, 200),  # cobalt
    (180,  50,  50),  # dark red
    (100, 200,  50),  # apple green
    ( 50, 200, 150),  # seafoam
    (200, 150,   0),  # dark yellow
    (100,  50, 100),  # plum
]


def _get_rich_color(class_id: int) -> Tuple[int, int, int]:
    """Return a stable distinct colour for *class_id* (1-indexed)."""
    if class_id == 0:
        return (20, 20, 20)  # background
    return _RICH_BASE_COLORS[(class_id - 1) % len(_RICH_BASE_COLORS)]


# ---------------------------------------------------------------------------
# Semantic-map colour palette
# ---------------------------------------------------------------------------
_SEMANTIC_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (30,  30,  30),   # background — dark grey
    1: (220, 50,  50),   # person     — red
    3: (50,  100, 220),  # car        — blue
}

_LEGEND_LABELS = {0: "background", 1: "person", 3: "car"}


# ---------------------------------------------------------------------------
# GT semantic map builder
# ---------------------------------------------------------------------------

def build_gt_semantic_map(
    anns: list,
    H: int,
    W: int,
    label_map: Dict[int, int],
) -> np.ndarray:
    """
    Convert instance-level annotations to a ``(H, W)`` semantic map.

    Each pixel is set to the COCO class-id of the annotation covering it.
    If annotations overlap, last-writer wins (all annotations of a given
    class contribute identically).
    """
    semantic = np.zeros((H, W), dtype=np.int32)
    for ann in anns:
        if ann.class_id not in label_map:
            continue
        mask = rletools.decode(ann.mask_rle).astype(bool)
        semantic[mask] = label_map[ann.class_id]
    return semantic


# ---------------------------------------------------------------------------
# SAM-based semantic predictor (pretrained or finetuned, GT-box prompted)
# ---------------------------------------------------------------------------

def sam_predict_semantic(
    model: SamModel,
    processor: SamProcessor,
    device: torch.device,
    image: Image.Image,
    anns: list,
    label_map: Dict[int, int],
    finetuned_protocol: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Prompt SAM with ground-truth bounding boxes and convert per-instance
    predictions into a semantic map by assigning each mask its GT class label.

    Parameters
    ----------
    finetuned_protocol : bool
        If True, use the finetuned inference protocol
        (batched boxes (B,1,N,4), multimask_output=False, F.interpolate).
        Required when using finetuned SAM weights to avoid near-zero masks.
        If False, use the pretrained protocol
        (per-object boxes (1,N,1,4), multimask_output=True, post_process_masks).

    Returns
    -------
    semantic_map : np.ndarray, shape (H, W), dtype int32
    inference_time : float
    """
    t0 = time.time()
    img_np = np.array(image)
    H, W  = img_np.shape[:2]
    semantic = np.zeros((H, W), dtype=np.int32)

    if not anns:
        return semantic, time.time() - t0

    # Filter to known classes
    valid_anns = [a for a in anns if a.class_id in label_map]
    if not valid_anns:
        return semantic, time.time() - t0

    boxes           = [ann.bbox_xyxy for ann in valid_anns]

    if finetuned_protocol:
        # ---- finetuned protocol: batched (1, 1, N, 4), multimask=False ----
        # Matches postprocess_preds_and_flatten() in sam_finetune.py exactly:
        #   256 → 1024 → crop padding → original size
        batched_boxes = [[boxes]]   # (1, 1, N, 4)
        inputs = processor(images=[img_np], input_boxes=batched_boxes, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        ib = inputs["input_boxes"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pv, input_boxes=ib, multimask_output=False)

        # pred_masks: (1, N, 1, 256, 256) → (N, 1, 256, 256)
        pred_masks_raw = outputs.pred_masks.view(
            -1, 1, outputs.pred_masks.shape[-2], outputs.pred_masks.shape[-1]
        ).cpu()  # (N, 1, 256, 256)

        # 3-step upsampling — identical to postprocess_preds_and_flatten
        orig_h, orig_w     = inputs["original_sizes"][0].tolist()
        reshaped_h, reshaped_w = inputs["reshaped_input_sizes"][0].tolist()

        up = F.interpolate(pred_masks_raw, size=(1024, 1024),
                           mode="bilinear", align_corners=False)
        up = up[..., :int(reshaped_h), :int(reshaped_w)]
        upscaled = F.interpolate(up, size=(int(orig_h), int(orig_w)),
                                 mode="bilinear", align_corners=False).squeeze(1)  # (N, H, W)

        scores_ft   = torch.sigmoid(upscaled).mean(dim=(-1, -2))   # (N,)
        pred_binary = (torch.sigmoid(upscaled) > 0.5).numpy()

        order = torch.argsort(scores_ft)   # ascending: best mask wins on overlaps
        for j in order.tolist():
            if j >= len(valid_anns):
                continue
            ann      = valid_anns[j]
            class_id = label_map[ann.class_id]
            semantic[pred_binary[j]] = class_id

    else:
        # ---- pretrained protocol: per-object (1, N, 1, 4), multimask=True ----
        input_boxes_fmt = [[[box] for box in boxes]]   # (1, N, 1, 4)

        inputs = processor(images=[img_np], input_boxes=input_boxes_fmt, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        ib = inputs["input_boxes"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pv, input_boxes=ib, multimask_output=True)

        iou_scores = outputs.iou_scores.cpu()[0]          # (N, 3)
        best_idx   = iou_scores.argmax(dim=-1)             # (N,)
        best_score = iou_scores[torch.arange(len(valid_anns)), best_idx]

        masks_pp  = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        masks_out = masks_pp[0]   # (N, 3, H, W) bool

        order = torch.argsort(best_score)   # ascending
        for j in order.tolist():
            ann      = valid_anns[j]
            class_id = label_map[ann.class_id]
            mask     = masks_out[j, int(best_idx[j])].numpy()
            semantic[mask] = class_id

    return semantic, time.time() - t0


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _semantic_to_rgb(semantic: np.ndarray) -> np.ndarray:
    """Convert semantic map to an RGB uint8 array (standard 2-class palette)."""
    rgb = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    for class_id, color in _SEMANTIC_PALETTE.items():
        rgb[semantic == class_id] = color
    # Any unrecognised id → magenta for debugging
    known = np.zeros(semantic.shape, dtype=bool)
    for cid in _SEMANTIC_PALETTE:
        known |= (semantic == cid)
    rgb[~known] = (255, 0, 255)
    return rgb


def _semantic_to_rgb_rich(
    semantic: np.ndarray,
    id_to_label: Dict[int, str],
) -> np.ndarray:
    """Convert open-vocabulary semantic map to RGB using the rich palette."""
    rgb = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    # background
    rgb[semantic == 0] = _get_rich_color(0)
    for class_id in id_to_label:
        rgb[semantic == class_id] = _get_rich_color(class_id)
    return rgb


def _blend(image: Image.Image, color_map: np.ndarray, alpha: float = 0.55) -> Image.Image:
    base = np.array(image).astype(float)
    overlay = color_map.astype(float)
    blended = (alpha * overlay + (1.0 - alpha) * base).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def _add_title_bar(img: Image.Image, title: str, bar_h: int = 38) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
        )
    except Exception:
        font = ImageFont.load_default()
    bbox   = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((w - text_w) // 2, (bar_h - (bbox[3] - bbox[1])) // 2),
              title, fill="black", font=font)
    canvas.paste(img, (0, bar_h))
    return canvas


def _add_legend(width: int) -> Image.Image:
    """Create a small legend strip for the standard 2-class palette."""
    bar_h  = 30
    n      = len(_LEGEND_LABELS)
    canvas = Image.new("RGB", (width, bar_h), (245, 245, 245))
    draw   = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
        )
    except Exception:
        font = ImageFont.load_default()

    block_w = width // n
    for idx, (cid, name) in enumerate(_LEGEND_LABELS.items()):
        x0 = idx * block_w
        color = _SEMANTIC_PALETTE.get(cid, (128, 128, 128))
        draw.rectangle([x0, 4, x0 + 20, bar_h - 4], fill=color)
        draw.text((x0 + 24, 7), name, fill="black", font=font)
    return canvas


def _add_legend_rich(
    width: int,
    id_to_label: Dict[int, str],
) -> Image.Image:
    """Dynamic legend strip for open-vocabulary results."""
    # Add background entry
    entries = {0: "background", **id_to_label}
    n       = len(entries)
    item_w  = max(120, width // max(n, 1))
    bar_h   = 28
    canvas_w = min(width, item_w * n)
    canvas  = Image.new("RGB", (canvas_w, bar_h), (245, 245, 245))
    draw    = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
        )
    except Exception:
        font = ImageFont.load_default()

    x_cursor = 0
    for cid, name in entries.items():
        if x_cursor + item_w > canvas_w:
            break
        color = _get_rich_color(cid)
        draw.rectangle([x_cursor, 4, x_cursor + 16, bar_h - 4], fill=color)
        draw.text((x_cursor + 20, 6), name[:12], fill="black", font=font)
        x_cursor += item_w
    return canvas


def create_viz(
    image: Image.Image,
    gt_semantic: np.ndarray,
    pred_semantic: np.ndarray,
    title_suffix: str = "",
) -> Image.Image:
    """Three-panel visualisation: original | GT semantic | predicted semantic."""
    gt_rgb   = _semantic_to_rgb(gt_semantic)
    pred_rgb = _semantic_to_rgb(pred_semantic)

    gt_blend   = _blend(image, gt_rgb,   alpha=0.60)
    pred_blend = _blend(image, pred_rgb, alpha=0.60)

    gt_panel   = _add_title_bar(gt_blend,      "GT Semantic")
    orig_panel = _add_title_bar(image.copy(),  "Original")
    pred_panel = _add_title_bar(pred_blend,    f"Predicted Semantic{title_suffix}")

    w, h = gt_panel.size
    legend = _add_legend(w * 3)
    canvas = Image.new("RGB", (w * 3, h + legend.size[1]), "white")
    canvas.paste(orig_panel, (0,     0))
    canvas.paste(gt_panel,   (w,     0))
    canvas.paste(pred_panel, (w * 2, 0))
    canvas.paste(legend,     (0, h))
    return canvas


def create_viz_rich(
    image: Image.Image,
    pred_semantic: np.ndarray,
    id_to_label: Dict[int, str],
    title_suffix: str = "",
) -> Image.Image:
    """Two-panel visualisation for open-vocabulary mode: original | predicted."""
    pred_rgb   = _semantic_to_rgb_rich(pred_semantic, id_to_label)
    pred_blend = _blend(image, pred_rgb, alpha=0.60)

    orig_panel = _add_title_bar(image.copy(),  "Original")
    pred_panel = _add_title_bar(pred_blend,    f"Open-Vocab Semantic{title_suffix}")

    w, h = orig_panel.size
    legend = _add_legend_rich(w * 2, id_to_label)

    # Legend may need extra rows if there are many classes — stack below panels
    canvas = Image.new("RGB", (w * 2, h + legend.size[1]), "white")
    canvas.paste(orig_panel, (0, 0))
    canvas.paste(pred_panel, (w, 0))
    canvas.paste(legend,     (0, h))
    return canvas


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic segmentation on KITTI-MOTS (Task h)"
    )
    parser.add_argument(
        "--mode",
        choices=["text_prompt", "pretrained_sam", "finetuned_sam", "rich_text_prompt"],
        required=True,
        help=(
            "text_prompt: zero-shot GroundedSAM (person+car); "
            "pretrained_sam: SAM (pretrained) with GT boxes; "
            "finetuned_sam: SAM (finetuned) with GT boxes; "
            "rich_text_prompt: open-vocabulary many-class qualitative demo."
        ),
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to finetuned SAM .pth checkpoint (required for finetuned_sam).",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/sam-vit-base",
        help="HuggingFace model id for SAM (pretrained/finetuned modes).",
    )
    parser.add_argument(
        "--dino_id",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace model id for GroundingDINO (text_prompt mode).",
    )
    parser.add_argument(
        "--rich_labels",
        type=str,
        default=RICH_LABELS,
        help="Period-separated labels for rich_text_prompt mode. Defaults to a large driving-scene set.",
    )
    parser.add_argument(
        "--text_labels",
        type=str,
        default="person. car.",
        help='Period-separated class labels, e.g. "person. car."',
    )
    parser.add_argument(
        "--root",
        type=str,
        default="~/mcv/datasets/C5/KITTI-MOTS/",
        help="Path to KITTI-MOTS dataset root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "dev", "validation"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images (for quick debugging).",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.35,
        help="GroundingDINO box confidence threshold.",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.25,
        help="GroundingDINO text confidence threshold.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "results_semantic"),
        help="Root directory for outputs.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name (subdirectory under output_dir). Auto-generated if omitted.",
    )
    parser.add_argument(
        "--save_viz_every",
        type=int,
        default=20,
        help="Save a visualisation every N frames.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- experiment directories ----------------------------------------
    exp_name = args.exp_name or f"semantic_{args.mode}_{args.split}"
    out_dir  = Path(args.output_dir) / exp_name
    viz_dir  = out_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ---- dataset -----------------------------------------------------------
    print(f"Loading KITTI-MOTS [{args.split}] …")
    ds = KITTIMOTS(
        root=args.root, split=args.split, ann_source="txt", compute_boxes=True
    )
    n_total = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"  {n_total} frames to process.")

    label_map = ds.LABELS_MAPPING   # KITTI id → COCO id

    # ---- load model --------------------------------------------------------
    device = torch.device(args.device)
    model_obj: Any = None

    if args.mode == "text_prompt":
        print("Mode: text_prompt (GroundedSAM zero-shot)")
        model_obj = GroundedSamWrapper(
            dino_model_id=args.dino_id,
            sam_model_id=args.model_id,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=str(device),
        )
        prompt_strategy = TextPromptStrategy(text_labels=args.text_labels)

    elif args.mode == "rich_text_prompt":
        print("Mode: rich_text_prompt (open-vocabulary, qualitative only)")
        model_obj = GroundedSamWrapper(
            dino_model_id=args.dino_id,
            sam_model_id=args.model_id,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=str(device),
        )
        prompt_strategy = TextPromptStrategy(text_labels=args.rich_labels)

    else:  # pretrained_sam or finetuned_sam
        mode_label = "finetuned SAM" if args.mode == "finetuned_sam" else "pretrained SAM"
        print(f"Mode: {args.mode} ({mode_label} with GT-box prompts)")
        processor = SamProcessor.from_pretrained(args.model_id)
        sam_model = SamModel.from_pretrained(args.model_id).to(device)

        if args.mode == "finetuned_sam":
            if not args.weights:
                raise ValueError("--weights is required for finetuned_sam mode.")
            ckpt       = torch.load(args.weights, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = sam_model.load_state_dict(state_dict, strict=False)
            print(
                f"  Loaded weights: {args.weights} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        sam_model.eval()
        model_obj = (sam_model, processor)

    # ---- inference loop ---------------------------------------------------
    evaluator      = SemanticEvaluator(class_ids=CLASS_IDS)
    total_inf_time = 0.0
    # Accumulate seen labels across frames (for rich mode legend summary)
    rich_global_id_to_label: Dict[int, str] = {}

    for i in tqdm(range(n_total), desc=f"[{args.mode}]"):
        image, anns, meta = ds[i]
        W, H = image.size

        # --- GT semantic map ---
        gt_semantic = build_gt_semantic_map(anns, H, W, label_map)

        # --- Predicted semantic map ---
        if args.mode == "text_prompt":
            prompt_data  = prompt_strategy.generate_prompt(image, anns)
            pred_semantic, inf_time = model_obj.predict_semantic(image, prompt_data)

        elif args.mode == "rich_text_prompt":
            prompt_data = prompt_strategy.generate_prompt(image, anns)
            pred_semantic, id_to_label, inf_time = model_obj.predict_semantic_open(
                image, prompt_data
            )
            rich_global_id_to_label.update(id_to_label)

        else:
            sam_model, processor = model_obj
            finetuned = (args.mode == "finetuned_sam")
            pred_semantic, inf_time = sam_predict_semantic(
                sam_model, processor, device, image, anns, label_map,
                finetuned_protocol=finetuned,
            )

        total_inf_time += inf_time

        # --- Update evaluator (only for modes with GT class alignment) ---
        if args.mode != "rich_text_prompt":
            evaluator.update(pred_semantic, gt_semantic)

        # --- Visualise every N frames ---
        if i % args.save_viz_every == 0:
            seq   = meta.get("seq",   "??")
            frame = meta.get("frame", "??")
            fname = f"{i:04d}_seq{seq}_frame{frame}.png"
            if args.mode == "rich_text_prompt":
                viz = create_viz_rich(
                    image, pred_semantic, id_to_label,
                    title_suffix=f" ({len(id_to_label)} classes)",
                )
            else:
                viz = create_viz(
                    image, gt_semantic, pred_semantic,
                    title_suffix=f" ({args.mode})",
                )
            viz.save(viz_dir / fname)

    # ---- compute and save metrics ----------------------------------------
    metrics_path = out_dir / "metrics.json"

    if args.mode == "rich_text_prompt":
        # No GT-based metrics; save the observed label inventory instead
        rich_summary = {
            "mode": "rich_text_prompt",
            "n_images": n_total,
            "n_distinct_labels_observed": len(rich_global_id_to_label),
            "labels_observed": dict(sorted(rich_global_id_to_label.items())),
            "text_labels_prompt": args.rich_labels,
            "performance/avg_fps":         n_total / total_inf_time if total_inf_time > 0 else 0.0,
            "performance/total_inf_time_s": total_inf_time,
            "performance/avg_latency_ms":   (total_inf_time / n_total * 1000) if n_total else 0.0,
        }
        with open(metrics_path, "w") as f:
            json.dump(rich_summary, f, indent=2)
        print(f"\n[rich_text_prompt] Observed {len(rich_global_id_to_label)} distinct classes:")
        for cid, name in sorted(rich_global_id_to_label.items()):
            print(f"  {cid:3d}: {name}")
        print(f"Label inventory saved to {metrics_path}")
        print(f"Visualisations saved to {viz_dir}")
        return

    metrics = evaluator.compute()
    avg_fps = n_total / total_inf_time if total_inf_time > 0 else 0.0
    metrics["performance/avg_fps"]            = avg_fps
    metrics["performance/total_inf_time_s"]   = total_inf_time
    metrics["performance/avg_latency_ms"]     = (total_inf_time / n_total * 1000) if n_total else 0.0

    print("\n========== Semantic Segmentation Results ==========")
    print(f"  Mode   : {args.mode}")
    print(f"  Split  : {args.split}  |  Frames: {n_total}")
    print(f"  mIoU   : {metrics.get('overall/mIoU', float('nan')):.4f}")
    for cid in CLASS_IDS:
        name = CLASS_NAMES.get(cid, str(cid))
        print(f"  {name:<8} IoU : {metrics.get(f'{name}/IoU', float('nan')):.4f}")
    print(f"  FPS    : {avg_fps:.2f}")
    print("===================================================\n")

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(f"Visualisations saved to {viz_dir}")


if __name__ == "__main__":
    main()
