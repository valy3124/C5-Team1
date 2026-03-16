"""
run_deart_inference.py

Task (f): Cross-domain evaluation — apply SAM (pre-trained and KITTI-MOTS
finetuned) to DEArt, an art-painting dataset with strong domain shift.

There are two complementary approaches:

  Approach 1 – Text prompts  (GroundedSAM)
  ──────────────────────────────────────────
  --mode text_prompt_pretrained
      GroundingDINO (pretrained) detects DeART objects from text labels;
      SAM (pretrained) segments them.  Evaluated with detection AP50 against
      GT bounding boxes.

  --mode text_prompt_finetuned
      Placeholder — requires a SAM model finetuned with text prompts on
      KITTI-MOTS. Pass --weights when available; currently raises an error
      with a clear message.

  Approach 2 – GT bounding-box prompts
  ──────────────────────────────────────────
  --mode gt_box_pretrained
      SAM (pretrained, facebook/sam-vit-base) segmented from GT boxes.

  --mode gt_box_finetuned
      SAM (KITTI-MOTS finetuned mask-decoder) segmented from GT boxes.
      Requires --weights pointing to a best_model.pth checkpoint.

Both GT-box approaches use the same evaluation:
  Since DeART has NO ground-truth segmentation masks (only bboxes), we use
  the filled GT bounding box as a proxy GT mask and report:
    • mask_box_iou  — IoU(predicted_mask, GT_box_rectangle)
    • mask_coverage — mask_pixels_inside_box / box_pixels
    • mask_precision — mask_pixels_inside_box / total_mask_pixels
  These cleanly capture how well SAM segments the annotated region.

Compared metrics reveal the domain-shift impact:
  pretrained SAM → generalises to art; finetuned SAM → may overfit to
  driving-scene appearance.

Text prompt labels for DeART (person super-class):
  "angel . centaur . crucifixion . devil . god the father . judith .
   knight . monk . nude . person . shepherd ."

Outputs (under --output_dir/<exp_name>/)
─────────────────────────────────────────
  metrics.json        — all computed metrics
  viz/NNNN_*.png      — side-by-side visualisations

Usage (from Week2/ directory)
─────────────────────────────
  # GT-box pretrained
  python -m src.inference.run_deart_inference --mode gt_box_pretrained

  # GT-box finetuned
  python -m src.inference.run_deart_inference \\
      --mode gt_box_finetuned \\
      --weights results_finetune/sam_bbox_j6cstc09/best_model.pth

  # Text-prompt pretrained
  python -m src.inference.run_deart_inference --mode text_prompt_pretrained
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import SamModel, SamProcessor
import pycocotools.mask as rletools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ── path bootstrap ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent   # Week2/
sys.path.insert(0, str(ROOT / "src"))

from datasets import DEART
from models.grounded_sam import GroundedSamWrapper
from prompting.text import TextPromptStrategy

# ── DeART text-prompt label set ─────────────────────────────────────────────
# All 11 human-like classes that DeART maps to "person"
DEART_TEXT_LABELS = (
    "angel . centaur . crucifixion . devil . god the father . judith . "
    "knight . monk . nude . person . shepherd ."
)

# Unique per-class IDs — must match DEART.LABELS_MAPPING in datasets.py
DEART_CLASS_MAP: Dict[str, int] = {
    "angel": 1, "centaur": 2, "crucifixion": 3, "devil": 4,
    "god the father": 5, "judith": 6, "knight": 7, "monk": 8,
    "nude": 9, "person": 10, "shepherd": 11,
}
DEART_ID_TO_NAME: Dict[int, str] = {v: k for k, v in DEART_CLASS_MAP.items()}
DEART_COCO_CATEGORIES = [{"id": cid, "name": name} for name, cid in DEART_CLASS_MAP.items()]

# ── colour palette ───────────────────────────────────────────────────────────
_BOX_COLOR    = (255, 80, 0)    # orange — GT boxes
_MASK_COLOR   = (50, 180, 255)  # blue   — predicted masks
_BG_ALPHA     = 0.45


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: box → binary rectangle mask
# ═══════════════════════════════════════════════════════════════════════════

def box_to_mask(bbox_xyxy: Tuple[int, int, int, int], H: int, W: int) -> np.ndarray:
    """Return a boolean ``(H, W)`` array with the bounding-box region filled."""
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H))
    m = np.zeros((H, W), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


# ═══════════════════════════════════════════════════════════════════════════
#  Per-instance mask quality metrics (proxy, no GT masks needed)
# ═══════════════════════════════════════════════════════════════════════════

def mask_box_metrics(pred_mask: np.ndarray, bbox_xyxy: Tuple, H: int, W: int) -> Dict:
    """
    Compute quality of *pred_mask* relative to the GT *bbox*.

    mask_box_iou  — IoU between predicted mask and box-filled rectangle
    mask_coverage — fraction of GT-box area covered by the mask
    mask_precision — fraction of mask pixels that lie inside the GT box
    """
    box_mask = box_to_mask(bbox_xyxy, H, W)
    pred     = pred_mask.astype(bool)

    inter = (pred & box_mask).sum()
    union = (pred | box_mask).sum()
    iou   = float(inter) / float(union + 1e-6)

    coverage  = float(inter) / float(box_mask.sum() + 1e-6)
    precision = float(inter) / float(pred.sum() + 1e-6)

    return {"mask_box_iou": iou, "mask_coverage": coverage, "mask_precision": precision}


# ═══════════════════════════════════════════════════════════════════════════
#  SAM inference helpers — mirrors run_semantic_segmentation.py
# ═══════════════════════════════════════════════════════════════════════════

def _sam_predict_instances_pretrained(
    model: SamModel,
    processor: SamProcessor,
    device: torch.device,
    img_np: np.ndarray,
    boxes: List[Tuple],
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run pretrained-protocol SAM on a list of boxes.

    Returns
    -------
    masks : list of (H, W) bool arrays, one per box
    scores : list of float IOU scores
    """
    if not boxes:
        return [], []

    # (1, N, 1, 4) format — per-object prompts, multimask=True, post_process_masks
    input_boxes_fmt = [[[box] for box in boxes]]
    inputs = processor(images=[img_np], input_boxes=input_boxes_fmt, return_tensors="pt")
    pv = inputs["pixel_values"].to(device)
    ib = inputs["input_boxes"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pv, input_boxes=ib, multimask_output=True)

    iou_scores = outputs.iou_scores.cpu()[0]      # (N, 3)
    best_idx   = iou_scores.argmax(dim=-1)         # (N,)
    best_score = iou_scores[torch.arange(len(boxes)), best_idx].tolist()

    masks_pp = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    masks_out = masks_pp[0]   # (N, 3, H, W) bool

    masks  = [masks_out[j, int(best_idx[j])].numpy() for j in range(len(boxes))]
    return masks, best_score


def _sam_predict_instances_finetuned(
    model: SamModel,
    processor: SamProcessor,
    device: torch.device,
    img_np: np.ndarray,
    boxes: List[Tuple],
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run finetuned-protocol SAM on a list of boxes.

    Mirrors prepare_batch_for_sam + postprocess_preds_and_flatten from
    sam_finetune.py exactly: (1, N, 4) → multimask=False → 3-step interpolate.

    Returns
    -------
    masks : list of (H, W) bool arrays, one per box
    scores : list of float mean-sigmoid scores
    """
    if not boxes:
        return [], []

    # Pad to homogeneous length (here N=N, single image, no padding needed)
    batched_boxes = [[list(boxes)]]   # shape hint: [[N × 4]]
    inputs = processor(images=[img_np], input_boxes=batched_boxes, return_tensors="pt")
    pv = inputs["pixel_values"].to(device)
    ib = inputs["input_boxes"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pv, input_boxes=ib, multimask_output=False)

    # pred_masks: (1, N, 1, 256, 256) → (N, 1, 256, 256)
    N  = len(boxes)
    pred_masks_raw = outputs.pred_masks.view(N, 1,
        outputs.pred_masks.shape[-2], outputs.pred_masks.shape[-1]).cpu()

    # 3-step upsampling (matches postprocess_preds_and_flatten)
    orig_h, orig_w     = inputs["original_sizes"][0].tolist()
    reshaped_h, reshaped_w = inputs["reshaped_input_sizes"][0].tolist()

    up = F.interpolate(pred_masks_raw, size=(1024, 1024), mode="bilinear", align_corners=False)
    up = up[..., :int(reshaped_h), :int(reshaped_w)]
    upscaled = F.interpolate(up, size=(int(orig_h), int(orig_w)),
                             mode="bilinear", align_corners=False).squeeze(1)   # (N, H, W)

    binary = (upscaled > 0).numpy()   # (N, H, W)  — matches training threshold
    scores = torch.sigmoid(upscaled).mean(dim=(-1, -2)).tolist()

    return [binary[j] for j in range(N)], scores


# ═══════════════════════════════════════════════════════════════════════════
#  Detection AP helper (for text-prompt mode)
# ═══════════════════════════════════════════════════════════════════════════

def _box_iou(a: Tuple, b: Tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def build_coco_det_gt(dataset: DEART) -> COCO:
    """Build a COCO-format GT object (bbox only) from a DEART split."""
    images, anns = [], []
    ann_id = 1
    for i in range(len(dataset)):
        img, inst_anns, meta = dataset[i]
        w, h = img.size
        images.append({"id": meta["index"], "width": w, "height": h})
        for ann in inst_anns:
            x1, y1, x2, y2 = ann.bbox_xyxy
            anns.append({
                "id": ann_id, "image_id": meta["index"],
                "category_id": ann.class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": max(0, (x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
                # segmentation not used by bbox eval but needed for COCO struct
                "segmentation": [],
            })
            ann_id += 1

    coco = COCO()
    coco.dataset = {
        "images": images,
        "annotations": anns,
        "categories": DEART_COCO_CATEGORIES,
    }
    coco.createIndex()
    return coco


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _add_title(img: Image.Image, title: str, bar_h: int = 34) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    tb = draw.textbbox((0, 0), title, font=font)
    draw.text(((w - (tb[2] - tb[0])) // 2, (bar_h - (tb[3] - tb[1])) // 2),
              title, fill="black", font=font)
    canvas.paste(img, (0, bar_h))
    return canvas


def _overlay_masks(
    image: Image.Image,
    masks: List[np.ndarray],
    boxes: List[Tuple],
    alpha: float = 0.50,
) -> Image.Image:
    """Overlay SAM masks (blue) and GT boxes (orange outline) onto image."""
    base  = np.array(image).astype(float)
    blend = base.copy()

    for mask in masks:
        color_layer = np.zeros_like(base)
        color_layer[mask.astype(bool)] = _MASK_COLOR
        blend = alpha * color_layer + (1 - alpha) * blend

    out = Image.fromarray(blend.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=_BOX_COLOR, width=2)
    return out


def create_viz_gtbox(
    image: Image.Image,
    gt_boxes: List[Tuple],
    pred_masks: List[np.ndarray],
    title: str = "",
) -> Image.Image:
    """Three-panel: original | GT boxes | GT boxes + predicted masks."""
    H, W = image.size[1], image.size[0]

    # Panel 1: original
    orig_panel = _add_title(image.copy(), "Original")

    # Panel 2: GT boxes only
    gt_img  = image.copy()
    draw_gt = ImageDraw.Draw(gt_img)
    for (x1, y1, x2, y2) in gt_boxes:
        draw_gt.rectangle([x1, y1, x2, y2], outline=_BOX_COLOR, width=2)
    gt_panel = _add_title(gt_img, "GT Boxes")

    # Panel 3: predicted masks + GT box outlines
    pred_panel = _add_title(
        _overlay_masks(image, pred_masks, gt_boxes),
        f"SAM Masks {title}",
    )

    w, h = orig_panel.size
    canvas = Image.new("RGB", (w * 3, h), "white")
    canvas.paste(orig_panel, (0,     0))
    canvas.paste(gt_panel,   (w,     0))
    canvas.paste(pred_panel, (w * 2, 0))
    return canvas


def create_viz_textprompt(
    image: Image.Image,
    gt_boxes: List[Tuple],
    det_boxes: List[Tuple],
    pred_masks: List[np.ndarray],
    title: str = "",
) -> Image.Image:
    """Three-panel: original | GT boxes | GroundedSAM detections + masks."""
    orig_panel  = _add_title(image.copy(), "Original")

    gt_img  = image.copy()
    draw_gt = ImageDraw.Draw(gt_img)
    for (x1, y1, x2, y2) in gt_boxes:
        draw_gt.rectangle([x1, y1, x2, y2], outline=_BOX_COLOR, width=2)
    gt_panel = _add_title(gt_img, "GT Boxes")

    det_img   = _overlay_masks(image, pred_masks, det_boxes, alpha=0.45)
    draw_det  = ImageDraw.Draw(det_img)
    for (x1, y1, x2, y2) in det_boxes:
        draw_det.rectangle([x1, y1, x2, y2], outline=(50, 255, 50), width=2)
    pred_panel = _add_title(det_img, f"GroundedSAM {title}")

    w, h = orig_panel.size
    canvas = Image.new("RGB", (w * 3, h), "white")
    canvas.paste(orig_panel, (0,     0))
    canvas.paste(gt_panel,   (w,     0))
    canvas.paste(pred_panel, (w * 2, 0))
    return canvas


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Task (f): SAM domain-shift evaluation on DEArt"
    )
    p.add_argument(
        "--mode",
        choices=[
            "gt_box_pretrained",
            "gt_box_finetuned",
            "text_prompt_pretrained",
            "text_prompt_finetuned",
        ],
        required=True,
    )
    p.add_argument(
        "--weights", type=str, default=None,
        help="Path to finetuned SAM checkpoint (.pth). Required for *_finetuned modes.",
    )
    p.add_argument(
        "--model_id", type=str, default="facebook/sam-vit-base",
        help="HuggingFace SAM model id.",
    )
    p.add_argument(
        "--dino_id", type=str, default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace GroundingDINO model id (text-prompt modes).",
    )
    p.add_argument(
        "--text_labels", type=str, default=DEART_TEXT_LABELS,
        help="Period-separated text labels for text-prompt modes.",
    )
    p.add_argument(
        "--box_threshold", type=float, default=0.30,
        help="GroundingDINO box confidence threshold.",
    )
    p.add_argument(
        "--text_threshold", type=float, default=0.25,
        help="GroundingDINO text threshold.",
    )
    p.add_argument(
        "--root", type=str,
        default=str(ROOT / "DEArt"),
        help="Path to the DeART dataset root directory.",
    )
    p.add_argument(
        "--split", type=str, default="validation",
        choices=["train", "dev", "validation", "train_full"],
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N images (quick debug).",
    )
    p.add_argument(
        "--output_dir", type=str,
        default=str(ROOT / "results_deart_NOCLASSMAP_NEWVIZ"),
    )
    p.add_argument(
        "--exp_name", type=str, default=None,
        help="Experiment subdirectory name (auto-generated if omitted).",
    )
    p.add_argument(
        "--save_viz_every", type=int, default=20,
        help="Save a visualisation every N frames.",
    )
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "text_prompt_finetuned" and not args.weights:
        raise ValueError(
            "[text_prompt_finetuned] --weights is required: provide the path to a "
            "SAM checkpoint finetuned with text-prompted boxes on KITTI-MOTS."
        )

    # ── directories ─────────────────────────────────────────────────────────
    exp_name = args.exp_name or f"deart_{args.mode}_{args.split}"
    out_dir  = Path(args.output_dir) / exp_name
    viz_dir  = out_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ── dataset ─────────────────────────────────────────────────────────────
    print(f"Loading DEArt [{args.split}] from {args.root} …")
    ds = DEART(root=args.root, split=args.split, ann_source="xml")
    n_total = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"  {n_total} images to process.")

    # Precompute a shared set of visualization indices so all modes write the
    # exact same images for qualitative comparison.
    viz_indices: set[int] = set()
    if args.save_viz_every > 0:
        viz_candidates: List[int] = []
        for i in range(n_total):
            _, anns, _ = ds[i]
            if anns:
                viz_candidates.append(i)
        viz_indices = set(viz_candidates[:: args.save_viz_every])
        print(
            f"  Viz policy: {len(viz_indices)} shared images "
            f"(every {args.save_viz_every} from GT-annotated frames)."
        )

    device = torch.device(args.device)

    # ── load model ──────────────────────────────────────────────────────────
    if args.mode in ("gt_box_pretrained", "gt_box_finetuned"):
        processor = SamProcessor.from_pretrained(args.model_id)
        sam_model = SamModel.from_pretrained(args.model_id).to(device)

        if args.mode == "gt_box_finetuned":
            if not args.weights:
                raise ValueError("--weights is required for gt_box_finetuned mode.")
            ckpt       = torch.load(args.weights, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = sam_model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded weights: {args.weights}  "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        sam_model.eval()
        is_finetuned = (args.mode == "gt_box_finetuned")

    else:  # text_prompt_pretrained / text_prompt_finetuned
        grounded_sam = GroundedSamWrapper(
            dino_model_id=args.dino_id,
            sam_model_id=args.model_id,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=str(device),
            label_to_class_id=DEART_CLASS_MAP,
        )
        if args.mode == "text_prompt_finetuned":
            ckpt = torch.load(args.weights, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = grounded_sam.sam_model.load_state_dict(
                state_dict, strict=False
            )
            grounded_sam.sam_model.eval()
            print(
                f"  Loaded finetuned SAM weights into GroundedSamWrapper: {args.weights} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        prompt_strategy = TextPromptStrategy(text_labels=args.text_labels)

    # ── metrics accumulators ─────────────────────────────────────────────────
    # GT-box modes
    all_iou:       List[float] = []
    all_coverage:  List[float] = []
    all_precision: List[float] = []
    all_scores:    List[float] = []
    # GT-box modes — per-class buckets
    per_class_iou:       Dict[int, List[float]] = defaultdict(list)
    per_class_coverage:  Dict[int, List[float]] = defaultdict(list)
    per_class_precision: Dict[int, List[float]] = defaultdict(list)

    # Text-prompt mode — COCO detection format
    coco_dt_list: List[Dict] = []
    coco_gt_full: Optional[COCO] = None
    # Box-count comparison accumulators (text modes)
    all_gt_box_counts:  List[int] = []
    all_det_box_counts: List[int] = []
    # Per-class box count accumulators (text modes)
    per_class_gt_counts:  Dict[int, int] = defaultdict(int)
    per_class_det_counts: Dict[int, int] = defaultdict(int)
    if args.mode in ("text_prompt_pretrained", "text_prompt_finetuned"):
        print("Pre-building COCO GT for detection AP …")
        coco_gt_full = build_coco_det_gt(
            DEART(root=args.root, split=args.split, ann_source="xml")
        )

    total_inf_time = 0.0

    # ── inference loop ───────────────────────────────────────────────────────
    for i in tqdm(range(n_total), desc=f"[{args.mode}]"):
        image, anns, meta = ds[i]
        W, H = image.size
        img_np = np.array(image)
        image_id = meta["index"]

        gt_boxes = [ann.bbox_xyxy for ann in anns]

        # ── GT-box modes ─────────────────────────────────────────────────
        if args.mode in ("gt_box_pretrained", "gt_box_finetuned"):
            if not gt_boxes:
                continue

            t0 = time.time()
            if is_finetuned:
                masks, scores = _sam_predict_instances_finetuned(
                    sam_model, processor, device, img_np, gt_boxes
                )
            else:
                masks, scores = _sam_predict_instances_pretrained(
                    sam_model, processor, device, img_np, gt_boxes
                )
            total_inf_time += time.time() - t0

            for ann, mask, score in zip(anns, masks, scores):
                box = ann.bbox_xyxy
                cid = ann.class_id
                m = mask_box_metrics(mask, box, H, W)
                all_iou.append(m["mask_box_iou"])
                all_coverage.append(m["mask_coverage"])
                all_precision.append(m["mask_precision"])
                all_scores.append(score)
                per_class_iou[cid].append(m["mask_box_iou"])
                per_class_coverage[cid].append(m["mask_coverage"])
                per_class_precision[cid].append(m["mask_precision"])

            if i in viz_indices:
                tag = "finetuned" if is_finetuned else "pretrained"
                viz = create_viz_gtbox(image, gt_boxes, masks, title=f"({tag})")
                viz.save(viz_dir / f"{i:04d}_img{image_id}.png")

        # ── text-prompt mode ──────────────────────────────────────────────
        else:
            prompt_data = prompt_strategy.generate_prompt(image, anns)
            t0 = time.time()
            masks_tensor, scores_tensor, inf_time = grounded_sam.predict(image, prompt_data)
            total_inf_time += time.time() - t0

            # masks_tensor: list of (1, N, 3, H, W) — unpack
            masks_out  = masks_tensor[0].squeeze(0)    # (N, 3, H, W)
            scores_out = scores_tensor.squeeze(0)      # (N, 3) or (N,)
            if scores_out.dim() == 1:
                scores_out = scores_out.unsqueeze(-1).expand(-1, 3)

            N_det = masks_out.shape[0]

            # Retrieve detected boxes from GroundingDINO directly
            det_boxes_raw, det_scores_raw, det_labels_raw = \
                grounded_sam._run_grounding_dino(image, args.text_labels)

            best_idx_per_det   = scores_out.argmax(dim=-1)
            best_score_per_det = scores_out[torch.arange(N_det), best_idx_per_det].tolist()

            # Collect predicted masks as binary arrays for visualisation
            pred_masks_viz = [
                masks_out[j, int(best_idx_per_det[j])].cpu().numpy()
                for j in range(N_det)
            ]

            # Track per-image box counts and per-class GT counts
            all_gt_box_counts.append(len(gt_boxes))
            all_det_box_counts.append(len(det_boxes_raw))
            for ann in anns:
                per_class_gt_counts[ann.class_id] += 1

            # Build COCO detection entries (bbox from predicted box)
            for j, (box, score_j) in enumerate(zip(det_boxes_raw, best_score_per_det)):
                x1, y1, x2, y2 = [float(v) for v in box]
                w_box = max(0.0, x2 - x1)
                h_box = max(0.0, y2 - y1)
                label_str = det_labels_raw[j] if j < len(det_labels_raw) else "person"
                cat_id = DEART_CLASS_MAP.get(label_str, 10)  # 10 = person as fallback
                coco_dt_list.append({
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, w_box, h_box],
                    "score": float(score_j),
                })
                per_class_det_counts[cat_id] += 1

            if i in viz_indices:
                det_boxes_viz = [(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                                 for b in det_boxes_raw]
                viz = create_viz_textprompt(
                    image, gt_boxes, det_boxes_viz, pred_masks_viz,
                    title="(pretrained)",
                )
                viz.save(viz_dir / f"{i:04d}_img{image_id}.png")

    # ── compute and save metrics ─────────────────────────────────────────────
    metrics: Dict = {"mode": args.mode, "split": args.split, "n_images": n_total}

    if args.mode in ("gt_box_pretrained", "gt_box_finetuned"):
        n_inst = len(all_iou)
        metrics.update({
            "n_instances": n_inst,
            "mask_box_iou/mean":     float(np.mean(all_iou))      if all_iou else 0.0,
            "mask_box_iou/std":      float(np.std(all_iou))       if all_iou else 0.0,
            "mask_box_iou/median":   float(np.median(all_iou))    if all_iou else 0.0,
            "mask_coverage/mean":    float(np.mean(all_coverage))  if all_coverage else 0.0,
            "mask_coverage/std":     float(np.std(all_coverage))   if all_coverage else 0.0,
            "mask_precision/mean":   float(np.mean(all_precision)) if all_precision else 0.0,
            "mask_precision/std":    float(np.std(all_precision))  if all_precision else 0.0,
            "model_score/mean":      float(np.mean(all_scores))    if all_scores else 0.0,
        })
        # Per-class breakdown
        per_class_out: Dict[str, Any] = {}
        for cat_id in sorted(DEART_ID_TO_NAME.keys()):
            cat_name = DEART_ID_TO_NAME[cat_id]
            ious  = per_class_iou[cat_id]
            covs  = per_class_coverage[cat_id]
            precs = per_class_precision[cat_id]
            per_class_out[cat_name] = {
                "n_instances":         len(ious),
                "mask_box_iou/mean":   float(np.mean(ious))  if ious  else 0.0,
                "mask_coverage/mean":  float(np.mean(covs))  if covs  else 0.0,
                "mask_precision/mean": float(np.mean(precs)) if precs else 0.0,
            }
        metrics["per_class"] = per_class_out

    else:  # text_prompt_pretrained / text_prompt_finetuned
        n_det = len(coco_dt_list)
        metrics["n_detections"] = n_det

        # ── box-count comparison metrics ──────────────────────────────────
        if all_gt_box_counts:
            total_gt  = sum(all_gt_box_counts)
            total_det = sum(all_det_box_counts)
            ratios = [
                d / g if g > 0 else float("inf")
                for g, d in zip(all_gt_box_counts, all_det_box_counts)
            ]
            finite_ratios = [r for r in ratios if r != float("inf")]
            metrics.update({
                # Aggregate counts
                "box_count/total_gt":             total_gt,
                "box_count/total_detected":        total_det,
                "box_count/total_det_minus_gt":    total_det - total_gt,
                # Per-image averages
                "box_count/avg_gt_per_image":      float(np.mean(all_gt_box_counts)),
                "box_count/avg_det_per_image":     float(np.mean(all_det_box_counts)),
                # det/gt ratio — tells over/under-detection per image
                "box_count/avg_det_gt_ratio":      float(np.mean(finite_ratios)) if finite_ratios else 0.0,
                "box_count/median_det_gt_ratio":   float(np.median(finite_ratios)) if finite_ratios else 0.0,
                # How many images are over/under/exact detected
                "box_count/pct_images_overdetected":  float(sum(d > g for g, d in zip(all_gt_box_counts, all_det_box_counts)) / len(all_gt_box_counts)),
                "box_count/pct_images_underdetected": float(sum(d < g for g, d in zip(all_gt_box_counts, all_det_box_counts)) / len(all_gt_box_counts)),
                "box_count/pct_images_exact":          float(sum(d == g for g, d in zip(all_gt_box_counts, all_det_box_counts)) / len(all_gt_box_counts)),
            })

        if n_det > 0 and coco_gt_full is not None:
            # Filter GT to only images we processed (first n_total)
            processed_ids = {ds[i][2]["index"] for i in range(n_total)}
            # Run COCO bbox evaluation
            try:
                coco_dt_obj = coco_gt_full.loadRes(coco_dt_list)
                coco_eval   = COCOeval(coco_gt_full, coco_dt_obj, "bbox")
                coco_eval.params.imgIds = sorted(processed_ids)
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                st = coco_eval.stats
                metrics.update({
                    "detection/AP":    float(st[0]),
                    "detection/AP50":  float(st[1]),
                    "detection/AP75":  float(st[2]),
                    "detection/APs":   float(st[3]),
                    "detection/APm":   float(st[4]),
                    "detection/APl":   float(st[5]),
                    "detection/AR1":   float(st[6]),
                    "detection/AR10":  float(st[7]),
                    "detection/AR100": float(st[8]),
                })
                # Per-class AP (separate COCOeval per category)
                per_class_out: Dict[str, Any] = {}
                for cat_id in sorted(DEART_ID_TO_NAME.keys()):
                    cat_name = DEART_ID_TO_NAME[cat_id]
                    try:
                        ev_cat = COCOeval(coco_gt_full, coco_dt_obj, "bbox")
                        ev_cat.params.catIds = [cat_id]
                        ev_cat.params.imgIds = sorted(processed_ids)
                        ev_cat.evaluate()
                        ev_cat.accumulate()
                        sc = ev_cat.stats
                        per_class_out[cat_name] = {
                            "AP":    float(sc[0]),
                            "AP50":  float(sc[1]),
                            "n_gt":  int(per_class_gt_counts.get(cat_id, 0)),
                            "n_det": int(per_class_det_counts.get(cat_id, 0)),
                        }
                    except Exception:
                        per_class_out[cat_name] = {
                            "AP": -1.0, "AP50": -1.0,
                            "n_gt":  int(per_class_gt_counts.get(cat_id, 0)),
                            "n_det": int(per_class_det_counts.get(cat_id, 0)),
                        }
                metrics["per_class"] = per_class_out
            except Exception as e:
                print(f"  COCO eval failed: {e}")

    # Performance stats
    if total_inf_time > 0:
        metrics["performance/avg_fps"]           = n_total / total_inf_time
        metrics["performance/total_inf_time_s"]  = total_inf_time
        metrics["performance/avg_latency_ms"]    = total_inf_time / n_total * 1000

    # ── print summary ───────────────────────────────────────────────────────
    print("\n========== DEArt Domain-Shift Results ==========")
    print(f"  Mode  : {args.mode}")
    print(f"  Split : {args.split}  |  Images: {n_total}")
    if args.mode in ("gt_box_pretrained", "gt_box_finetuned"):
        print(f"  mask_box_iou  (mean ± std) : "
              f"{metrics.get('mask_box_iou/mean', 0):.4f} ± "
              f"{metrics.get('mask_box_iou/std',  0):.4f}")
        print(f"  mask_coverage (mean ± std) : "
              f"{metrics.get('mask_coverage/mean', 0):.4f} ± "
              f"{metrics.get('mask_coverage/std',  0):.4f}")
        print(f"  mask_precision(mean ± std) : "
              f"{metrics.get('mask_precision/mean', 0):.4f} ± "
              f"{metrics.get('mask_precision/std',  0):.4f}")
        print(f"\n  Per-class mask_box_iou / coverage / precision:")
        for cat_name, cm in metrics.get("per_class", {}).items():
            n = cm["n_instances"]
            if n > 0:
                print(f"    {cat_name:20s}: IoU={cm['mask_box_iou/mean']:.3f}  "
                      f"cov={cm['mask_coverage/mean']:.3f}  "
                      f"prec={cm['mask_precision/mean']:.3f}  n={n}")
    else:
        print(f"  detection AP50 : {metrics.get('detection/AP50', float('nan')):.4f}")
        print(f"  detection AP   : {metrics.get('detection/AP',   float('nan')):.4f}")
        print(f"  n_detections   : {metrics.get('n_detections', 0)}")
        print(f"  box count — GT total / Det total : "
              f"{metrics.get('box_count/total_gt', 'N/A')} / "
              f"{metrics.get('box_count/total_detected', 'N/A')}")
        print(f"  avg det/gt ratio (per image)     : "
              f"{metrics.get('box_count/avg_det_gt_ratio', float('nan')):.3f}")
        print(f"  over-detected images : "
              f"{metrics.get('box_count/pct_images_overdetected', 0)*100:.1f}%  "
              f"under-detected: "
              f"{metrics.get('box_count/pct_images_underdetected', 0)*100:.1f}%")
        if "per_class" in metrics:
            print(f"\n  Per-class AP50 | n_gt | n_det:")
            for cat_name, cm in metrics["per_class"].items():
                n_gt  = cm.get("n_gt",  0)
                n_det = cm.get("n_det", 0)
                if n_gt > 0 or n_det > 0:
                    ap50 = cm.get("AP50", -1.0)
                    print(f"    {cat_name:20s}: AP50={ap50:6.3f}  "
                          f"GT={n_gt:4d}  Det={n_det:4d}")
    if "performance/avg_fps" in metrics:
        print(f"  FPS            : {metrics['performance/avg_fps']:.2f}")
    print("==================================================\n")

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(f"Visualisations saved to {viz_dir}")


if __name__ == "__main__":
    main()
