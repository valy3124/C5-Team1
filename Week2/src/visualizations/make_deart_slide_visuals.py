from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import SamModel, SamProcessor


ROOT = Path(__file__).resolve().parent.parent.parent
RES_ROOT = ROOT / "results_deart_NOCLASSMAP"
OUT_DIR = RES_ROOT / "slide_assets"

sys.path.insert(0, str(ROOT / "src"))
from datasets import DEART
from models.grounded_sam import GroundedSamWrapper


DEART_TEXT_LABELS = (
    "angel . centaur . crucifixion . devil . god the father . judith . "
    "knight . monk . nude . person . shepherd ."
)

DEART_CLASS_MAP: Dict[str, int] = {
    "angel": 1,
    "centaur": 2,
    "crucifixion": 3,
    "devil": 4,
    "god the father": 5,
    "judith": 6,
    "knight": 7,
    "monk": 8,
    "nude": 9,
    "person": 10,
    "shepherd": 11,
}

# Pick which qualifying sample to show in text prompt pipeline visuals.
# 0 = first valid sample, 1 = second valid sample, etc.
TEXT_PIPELINE_MATCH_RANK = 1


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    try:
        if bold:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _read_metrics(exp_name: str) -> Dict:
    p = RES_ROOT / exp_name / "metrics.json"
    with open(p, "r") as f:
        return json.load(f)


def make_metrics_summary() -> Path:
    gt_pre = _read_metrics("deart_gt_box_pretrained_validation")
    gt_fin = _read_metrics("deart_gt_box_finetuned_validation")
    tx_pre = _read_metrics("deart_text_prompt_pretrained_validation")
    tx_fin = _read_metrics("deart_text_prompt_finetuned_validation")

    w, h = 1800, 1050
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    title_f = _font(46, bold=True)
    sub_f = _font(30, bold=True)
    body_f = _font(24)
    mono_f = _font(25, bold=False)

    d.text((60, 35), "DeART Domain Shift - New Metrics Summary", fill="black", font=title_f)

    # Blocks
    left = (60, 120, 860, 990)
    right = (930, 120, 1740, 990)
    d.rounded_rectangle(left, outline=(40, 110, 180), width=4, radius=20)
    d.rounded_rectangle(right, outline=(180, 100, 30), width=4, radius=20)

    d.text((85, 145), "Mask quality from GT-box prompts", fill=(40, 110, 180), font=sub_f)
    d.text((960, 145), "Detection + count from text prompts", fill=(180, 100, 30), font=sub_f)

    y = 210
    d.text((95, y), "Pretrained SAM:", fill="black", font=body_f)
    y += 38
    d.text((120, y), f"mask_box_iou      = {gt_pre.get('mask_box_iou/mean', 0.0):.4f}", fill="black", font=mono_f)
    y += 34
    d.text((120, y), f"mask_coverage     = {gt_pre.get('mask_coverage/mean', 0.0):.4f}", fill="black", font=mono_f)
    y += 34
    d.text((120, y), f"mask_precision    = {gt_pre.get('mask_precision/mean', 0.0):.4f}", fill="black", font=mono_f)

    y += 60
    d.text((95, y), "KITTI-finetuned SAM:", fill="black", font=body_f)
    y += 38
    d.text((120, y), f"mask_box_iou      = {gt_fin.get('mask_box_iou/mean', 0.0):.4f}", fill="black", font=mono_f)
    y += 34
    d.text((120, y), f"mask_coverage     = {gt_fin.get('mask_coverage/mean', 0.0):.4f}", fill="black", font=mono_f)
    y += 34
    d.text((120, y), f"mask_precision    = {gt_fin.get('mask_precision/mean', 0.0):.4f}", fill="black", font=mono_f)

    y += 65
    delta_iou = gt_pre.get("mask_box_iou/mean", 0.0) - gt_fin.get("mask_box_iou/mean", 0.0)
    d.text((95, y), f"Delta IoU (pre - finetuned): {delta_iou:+.4f}", fill=(20, 90, 30), font=sub_f)

    y2 = 210
    d.text((970, y2), "Pretrained GroundedSAM:", fill="black", font=body_f)
    y2 += 38
    d.text((995, y2), f"AP50                = {tx_pre.get('detection/AP50', 0.0):.4f}", fill="black", font=mono_f)
    y2 += 34
    d.text((995, y2), f"AP                  = {tx_pre.get('detection/AP', 0.0):.4f}", fill="black", font=mono_f)
    y2 += 34
    d.text((995, y2), f"Det/GT total boxes  = {tx_pre.get('box_count/total_detected', 0)} / {tx_pre.get('box_count/total_gt', 0)}", fill="black", font=mono_f)
    y2 += 34
    d.text((995, y2), f"Avg det/gt ratio    = {tx_pre.get('box_count/avg_det_gt_ratio', 0.0):.3f}", fill="black", font=mono_f)
    y2 += 34
    d.text((995, y2), f"Over-detected imgs  = {100.0 * tx_pre.get('box_count/pct_images_overdetected', 0.0):.1f}%", fill="black", font=mono_f)

    y2 += 60
    d.text((970, y2), "Text-finetuned GroundedSAM:", fill="black", font=body_f)
    y2 += 38
    d.text((995, y2), f"AP50                = {tx_fin.get('detection/AP50', 0.0):.4f}", fill="black", font=mono_f)
    y2 += 34
    d.text((995, y2), f"AP                  = {tx_fin.get('detection/AP', 0.0):.4f}", fill="black", font=mono_f)
    y2 += 34
    d.text((995, y2), f"Det/GT total boxes  = {tx_fin.get('box_count/total_detected', 0)} / {tx_fin.get('box_count/total_gt', 0)}", fill="black", font=mono_f)
    y2 += 34
    d.text((995, y2), f"Avg det/gt ratio    = {tx_fin.get('box_count/avg_det_gt_ratio', 0.0):.3f}", fill="black", font=mono_f)
    y2 += 34
    d.text((995, y2), f"Over-detected imgs  = {100.0 * tx_fin.get('box_count/pct_images_overdetected', 0.0):.1f}%", fill="black", font=mono_f)

    y2 += 65
    d.text((970, y2), "Takeaway: text prompts over-detect in paintings.", fill=(140, 60, 15), font=sub_f)

    out = OUT_DIR / "metrics_summary.png"
    img.save(out)
    return out


def make_mask_iou_explainer() -> Path:
    w, h = 1800, 980
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    title_f = _font(46, bold=True)
    sub_f = _font(28, bold=True)
    body_f = _font(24)
    eq_f = _font(34, bold=True)

    d.text((60, 35), "What mask_box_iou means", fill="black", font=title_f)

    # Draw a toy scene on left
    panel = (70, 130, 860, 860)
    d.rounded_rectangle(panel, outline=(80, 80, 80), width=3, radius=16)
    x0, y0, x1, y1 = panel

    gt = (220, 260, 700, 720)
    pred_poly = [(170, 300), (680, 220), (760, 630), (300, 780)]

    # Union-ish background
    d.rectangle(gt, outline=(255, 120, 0), width=5)
    d.polygon(pred_poly, outline=(40, 130, 255), fill=(150, 200, 255), width=4)

    # Approx intersection overlay in green
    inter = (260, 290, 690, 700)
    d.rectangle(inter, fill=(100, 205, 100), outline=(20, 120, 20), width=3)

    d.text((120, 800), "Orange: GT box rectangle", fill=(220, 110, 0), font=body_f)
    d.text((430, 800), "Blue: predicted mask", fill=(35, 100, 220), font=body_f)

    # Right explanation
    rx = 930
    d.text((rx, 160), "Metric definition", fill=(30, 30, 30), font=sub_f)
    d.text((rx, 220), "1) Convert GT box to binary rectangle mask", fill="black", font=body_f)
    d.text((rx, 265), "2) Convert predicted mask to binary map", fill="black", font=body_f)
    d.text((rx, 310), "3) Compute Intersection and Union pixel counts", fill="black", font=body_f)
    d.text((rx, 355), "4) IoU = Intersection / Union", fill="black", font=body_f)

    d.rounded_rectangle((rx, 420, 1710, 560), outline=(35, 120, 35), width=4, radius=16)
    d.text((rx + 28, 470), "mask_box_iou = |pred_mask AND gt_box_mask| / |pred_mask OR gt_box_mask|", fill=(20, 95, 20), font=eq_f)

    d.text((rx, 620), "Interpretation", fill=(30, 30, 30), font=sub_f)
    d.text((rx, 675), "- High IoU: predicted shape aligns with annotated region", fill="black", font=body_f)
    d.text((rx, 715), "- Low IoU: mask misses box area or leaks outside", fill="black", font=body_f)
    d.text((rx, 755), "- Used here because DeART has boxes but no GT segmentation masks", fill="black", font=body_f)

    out = OUT_DIR / "mask_box_iou_explainer.png"
    img.save(out)
    return out


def _box_to_mask(bbox_xyxy: Tuple[int, int, int, int], h: int, w: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h))
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _predict_mask_pretrained(image_np: np.ndarray, box: Tuple[int, int, int, int], device: torch.device) -> np.ndarray:
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    model.eval()

    input_boxes_fmt = [[[box]]]
    inputs = processor(images=[image_np], input_boxes=input_boxes_fmt, return_tensors="pt")
    pv = inputs["pixel_values"].to(device)
    ib = inputs["input_boxes"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pv, input_boxes=ib, multimask_output=True)

    iou_scores = outputs.iou_scores.cpu()[0, 0]  # (3,)
    best_idx = int(iou_scores.argmax().item())

    masks_pp = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    return masks_pp[0][0, best_idx].numpy().astype(bool)


def _predict_masks_pretrained_many(
    image_np: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    device: torch.device,
) -> List[np.ndarray]:
    if not boxes:
        return []

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    model.eval()

    input_boxes_fmt = [[[box] for box in boxes]]
    inputs = processor(images=[image_np], input_boxes=input_boxes_fmt, return_tensors="pt")
    pv = inputs["pixel_values"].to(device)
    ib = inputs["input_boxes"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pv, input_boxes=ib, multimask_output=True)

    iou_scores = outputs.iou_scores.cpu()[0]
    best_idx = iou_scores.argmax(dim=-1)

    masks_pp = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    masks_out = masks_pp[0]
    return [masks_out[j, int(best_idx[j])].numpy().astype(bool) for j in range(len(boxes))]


def _overlay_boxes(image: Image.Image, boxes: List[Tuple[int, int, int, int]], color: Tuple[int, int, int], width: int = 4) -> Image.Image:
    out = image.copy()
    dr = ImageDraw.Draw(out)
    for (x1, y1, x2, y2) in boxes:
        dr.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=color, width=width)
    return out


def _overlay_masks_and_boxes(
    image: Image.Image,
    masks: List[np.ndarray],
    boxes: List[Tuple[int, int, int, int]],
    mask_color: Tuple[int, int, int] = (50, 180, 255),
    box_color: Tuple[int, int, int] = (255, 120, 0),
    alpha: float = 0.45,
) -> Image.Image:
    base = np.array(image).astype(float)
    blend = base.copy()
    for m in masks:
        layer = np.zeros_like(base)
        layer[m.astype(bool)] = mask_color
        blend = alpha * layer + (1.0 - alpha) * blend
    out = Image.fromarray(blend.clip(0, 255).astype(np.uint8))
    return _overlay_boxes(out, boxes, color=box_color, width=4)


def _resize_to_panel(im: Image.Image, panel_w: int) -> Image.Image:
    h_new = int(im.height * panel_w / im.width)
    return im.resize((panel_w, h_new), Image.Resampling.LANCZOS)


def make_text_prompt_pipeline_example() -> Path:
    """Create a step-by-step GroundedSAM pipeline visual for slides."""
    ds = DEART(root=str(ROOT / "DEArt"), split="validation", ann_source="xml")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    grounded_sam = GroundedSamWrapper(
        dino_model_id="IDEA-Research/grounding-dino-tiny",
        sam_model_id="facebook/sam-vit-base",
        box_threshold=0.30,
        text_threshold=0.25,
        device=device_str,
        label_to_class_id=DEART_CLASS_MAP,
    )

    chosen = None
    det_boxes: List[Tuple[int, int, int, int]] = []
    det_labels: List[str] = []
    valid_seen = 0
    for i in range(min(120, len(ds))):
        img, anns, meta = ds[i]
        boxes_raw, _scores_raw, labels_raw = grounded_sam._run_grounding_dino(img, DEART_TEXT_LABELS)
        if len(anns) >= 1 and 2 <= len(boxes_raw) <= 8:
            if valid_seen == TEXT_PIPELINE_MATCH_RANK:
                chosen = (img, anns, meta)
                det_boxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes_raw]
                det_labels = labels_raw
                break
            valid_seen += 1

    if chosen is None:
        img, anns, meta = ds[0]
        boxes_raw, _scores_raw, labels_raw = grounded_sam._run_grounding_dino(img, DEART_TEXT_LABELS)
        det_boxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes_raw]
        det_labels = labels_raw
    else:
        img, anns, meta = chosen

    img_np = np.array(img)
    seg_boxes = det_boxes[: min(6, len(det_boxes))]
    seg_masks = _predict_masks_pretrained_many(img_np, seg_boxes, torch.device(device_str))

    p1 = _resize_to_panel(img, 620)
    p2 = _resize_to_panel(_overlay_boxes(img, det_boxes, color=(40, 230, 40), width=4), 620)
    p3 = _resize_to_panel(_overlay_masks_and_boxes(img, seg_masks, seg_boxes), 620)

    title_h = 170
    label_h = 90
    canvas_h = title_h + p1.height + label_h + 30
    canvas_w = 3 * 620 + 60
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    d = ImageDraw.Draw(canvas)

    title_f = _font(40, bold=True)
    sub_f = _font(21)
    cap_f = _font(24, bold=True)
    txt_f = _font(20)

    d.text((20, 20), "Text-Prompt Pipeline (GroundingDINO -> SAM)", fill="black", font=title_f)
    d.text((20, 74), f"Image index: {meta.get('index', 0)} | GT boxes: {len(anns)} | DINO boxes: {len(det_boxes)}", fill=(40, 40, 40), font=sub_f)
    d.text((20, 104), "Prompt: angel . centaur . crucifixion . devil . god the father . judith . knight . monk . nude . person . shepherd .", fill=(40, 40, 40), font=txt_f)

    x = 10
    y_img = title_h
    panels = [
        (p1, "Step 1. Input image"),
        (p2, "Step 2. GroundingDINO detections"),
        (p3, "Step 3. SAM masks from DINO boxes"),
    ]
    for im_panel, label in panels:
        canvas.paste(im_panel, (x, y_img))
        d.text((x + 8, y_img + p1.height + 22), label, fill=(35, 35, 35), font=cap_f)
        x += 630

    unique_labels = sorted(set(lbl.strip().lower() for lbl in det_labels if lbl.strip()))
    shown_labels = ", ".join(unique_labels[:8]) if unique_labels else "none"
    d.text((20, canvas_h - 32), f"Detected label words (subset): {shown_labels}", fill=(80, 80, 80), font=txt_f)

    out = OUT_DIR / "text_prompt_pipeline_example.png"
    canvas.save(out)
    return out


def make_gt_box_pipeline_example() -> Path:
    """Create a step-by-step GT-box prompted SAM pipeline visual for slides."""
    ds = DEART(root=str(ROOT / "DEArt"), split="validation", ann_source="xml")
    img, anns, meta = ds[0]
    if not anns:
        for i in range(1, len(ds)):
            img, anns, meta = ds[i]
            if anns:
                break
    if not anns:
        raise RuntimeError("No DEArt annotations found for GT-box pipeline example.")

    boxes = [tuple(int(v) for v in a.bbox_xyxy) for a in anns]
    used_boxes = boxes[: min(8, len(boxes))]
    img_np = np.array(img)
    masks = _predict_masks_pretrained_many(
        img_np,
        used_boxes,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    p1 = _resize_to_panel(img, 620)
    p2 = _resize_to_panel(_overlay_boxes(img, used_boxes, color=(255, 120, 0), width=4), 620)
    p3 = _resize_to_panel(_overlay_masks_and_boxes(img, masks, used_boxes), 620)

    title_h = 150
    label_h = 90
    canvas_h = title_h + p1.height + label_h + 20
    canvas_w = 3 * 620 + 60
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    d = ImageDraw.Draw(canvas)

    title_f = _font(40, bold=True)
    sub_f = _font(22)
    cap_f = _font(24, bold=True)
    txt_f = _font(20)

    d.text((20, 20), "GT-Box Prompt Pipeline (SAM)", fill="black", font=title_f)
    d.text((20, 78), f"Image index: {meta.get('index', 0)} | GT boxes used as prompts: {len(used_boxes)}", fill=(40, 40, 40), font=sub_f)

    x = 10
    y_img = title_h
    panels = [
        (p1, "Step 1. Input image"),
        (p2, "Step 2. GT boxes from XML annotations"),
        (p3, "Step 3. SAM masks from GT boxes"),
    ]
    for im_panel, label in panels:
        canvas.paste(im_panel, (x, y_img))
        d.text((x + 8, y_img + p1.height + 20), label, fill=(35, 35, 35), font=cap_f)
        x += 630

    d.text((20, canvas_h - 30), "This mode isolates segmentation quality because box prompts come from ground truth.", fill=(80, 80, 80), font=txt_f)

    out = OUT_DIR / "gt_box_pipeline_example.png"
    canvas.save(out)
    return out


def make_mask_iou_real_example() -> Path:
    ds = DEART(root=str(ROOT / "DEArt"), split="validation", ann_source="xml")
    img, anns, meta = ds[0]
    if not anns:
        for i in range(1, len(ds)):
            img, anns, meta = ds[i]
            if anns:
                break
    if not anns:
        raise RuntimeError("No DEArt annotations found for real mask_iou example.")

    # Use the largest GT box in the chosen image to make the visual clearer.
    areas = [max(0, (a.bbox_xyxy[2] - a.bbox_xyxy[0]) * (a.bbox_xyxy[3] - a.bbox_xyxy[1])) for a in anns]
    k = int(np.argmax(areas))
    box = anns[k].bbox_xyxy

    img_np = np.array(img)
    h, w = img_np.shape[:2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_mask = _predict_mask_pretrained(img_np, box, device)
    box_mask = _box_to_mask(box, h, w)

    inter = pred_mask & box_mask
    union = pred_mask | box_mask
    iou = float(inter.sum()) / float(union.sum() + 1e-6)

    base = img_np.copy()

    def overlay(mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
        out = base.astype(float).copy()
        out[mask] = (1.0 - alpha) * out[mask] + alpha * np.array(color, dtype=float)
        return out.astype(np.uint8)

    gt_box_view = overlay(box_mask, (255, 150, 60), alpha=0.40)
    pred_view = overlay(pred_mask, (60, 150, 255), alpha=0.40)
    inter_view = overlay(inter, (80, 210, 80), alpha=0.55)

    # Draw GT rectangle for reference on all views.
    bx1, by1, bx2, by2 = [int(v) for v in box]
    for arr in (gt_box_view, pred_view, inter_view):
        dr = ImageDraw.Draw(Image.fromarray(arr))
        dr.rectangle([bx1, by1, bx2, by2], outline=(255, 120, 0), width=4)

    p1 = img.resize((640, int(img.height * 640 / img.width)), Image.Resampling.LANCZOS)
    p2 = Image.fromarray(gt_box_view).resize((640, int(img.height * 640 / img.width)), Image.Resampling.LANCZOS)
    p3 = Image.fromarray(pred_view).resize((640, int(img.height * 640 / img.width)), Image.Resampling.LANCZOS)
    p4 = Image.fromarray(inter_view).resize((640, int(img.height * 640 / img.width)), Image.Resampling.LANCZOS)

    title_h = 130
    row_h = p1.height + 90
    canvas = Image.new("RGB", (4 * 640 + 70, title_h + row_h), "white")
    d = ImageDraw.Draw(canvas)
    title_f = _font(40, bold=True)
    sub_f = _font(24, bold=True)
    txt_f = _font(22)

    d.text((20, 25), "mask_box_iou on a real DeART image", fill="black", font=title_f)
    d.text((20, 75), f"Image index: {meta.get('index', 0)} | class_id: {anns[k].class_id} | IoU = {iou:.4f}", fill=(20, 100, 20), font=sub_f)

    x = 10
    panels = [
        (p1, "Original image"),
        (p2, "GT box mask (orange)"),
        (p3, "Predicted mask (blue)"),
        (p4, "Intersection (green)"),
    ]
    for panel_img, label in panels:
        canvas.paste(panel_img, (x, title_h))
        d.text((x + 8, title_h + p1.height + 20), label, fill=(50, 50, 50), font=txt_f)
        x += 650

    out = OUT_DIR / "mask_box_iou_real_deart_example.png"
    canvas.save(out)
    return out


def _boost_gt_boxes(panel_arr: np.ndarray, thickness: int = 5) -> np.ndarray:
    """Find orange GT-box outline pixels and replace with thick bright red."""
    arr = panel_arr.copy()
    orange = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 150) & (arr[:, :, 2] < 80)
    # Manual binary dilation (no scipy dependency)
    dilated = orange.copy()
    for dy in range(-thickness, thickness + 1):
        for dx in range(-thickness, thickness + 1):
            dilated |= np.roll(np.roll(orange, dy, axis=0), dx, axis=1)
    arr[dilated] = [255, 0, 0]
    return arr


def _apply_gt_box_boost(im: Image.Image) -> Image.Image:
    """Split 3-panel viz image, boost GT boxes in middle panel, reassemble."""
    arr = np.array(im)
    w3 = arr.shape[1] // 3
    left   = arr[:, :w3, :]
    mid    = arr[:, w3:2*w3, :]
    right  = arr[:, 2*w3:, :]
    mid_boosted = _boost_gt_boxes(mid)
    return Image.fromarray(np.concatenate([left, mid_boosted, right], axis=1))


def _green_score(p: Path) -> int:
    """Proxy score for over-detection.

    It counts bright-green outline pixels in the right panel of existing
    visualizations (green boxes = predicted text-prompt detections).
    Higher score usually means more predicted boxes, but it is not a true
    detection metric and depends on line width/overlap/image size.
    """
    arr = np.array(Image.open(p).convert("RGB"))
    h, w = arr.shape[:2]
    right = arr[:, (2 * w) // 3 :, :]
    mask = (right[:, :, 1] > 220) & (right[:, :, 0] < 120) & (right[:, :, 2] < 120)
    return int(mask.sum())


def make_overdet_examples() -> Path:
    viz_dir = RES_ROOT / "deart_text_prompt_pretrained_validation" / "viz"
    imgs = sorted(viz_dir.glob("*.png"))
    if not imgs:
        raise FileNotFoundError(f"No images found in {viz_dir}")

    ranked = sorted(imgs, key=_green_score, reverse=True)
    top2 = [ranked[4], ranked[5]]  # ranks 5 & 6

    im1 = _apply_gt_box_boost(Image.open(top2[0]).convert("RGB"))
    im2 = _apply_gt_box_boost(Image.open(top2[1]).convert("RGB"))

    # Resize to a consistent width
    target_w = 1500
    im1 = im1.resize((target_w, int(im1.height * target_w / im1.width)), Image.Resampling.LANCZOS)
    im2 = im2.resize((target_w, int(im2.height * target_w / im2.width)), Image.Resampling.LANCZOS)

    pad = 30
    header_h = 120
    caption_h = 140
    h = header_h + im1.height + caption_h + pad + im2.height + caption_h + 50
    w = target_w + 2 * pad

    canvas = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(canvas)
    title_f = _font(42, bold=True)
    cap_f = _font(24)

    d.text((pad, 35), "Text-prompt over-detection examples (GT boxes vs predicted boxes)", fill="black", font=title_f)

    y = header_h
    canvas.paste(im1, (pad, y))
    y += im1.height
    d.text((pad, y + 12), f"Example 1: {top2[0].name} | green-box pixel score = {_green_score(top2[0])}", fill=(60, 60, 60), font=cap_f)
    y += caption_h

    canvas.paste(im2, (pad, y))
    y += im2.height
    d.text((pad, y + 12), f"Example 2: {top2[1].name} | green-box pixel score = {_green_score(top2[1])}", fill=(60, 60, 60), font=cap_f)

    out = OUT_DIR / "text_overdetection_examples.png"
    canvas.save(out)
    return out


def _draw_dino_boxes(
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    labels: List[str],
    target_label: str,
) -> Image.Image:
    out = image.copy()
    dr = ImageDraw.Draw(out)
    t = target_label.strip().lower()
    for box, lbl in zip(boxes, labels):
        lbl_norm = lbl.strip().lower().rstrip(".")
        is_target = lbl_norm == t
        color = (235, 45, 45) if is_target else (40, 230, 40)
        width = 4 if is_target else 2
        x1, y1, x2, y2 = [int(v) for v in box]
        dr.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return out


def make_dino_class_examples(target_label: str, top_k: int = 2, scan_limit: int = 300) -> Path:
    """Save top-k images where GroundingDINO predicts many boxes for a target label."""
    ds = DEART(root=str(ROOT / "DEArt"), split="validation", ann_source="xml")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    grounded_sam = GroundedSamWrapper(
        dino_model_id="IDEA-Research/grounding-dino-tiny",
        sam_model_id="facebook/sam-vit-base",
        box_threshold=0.30,
        text_threshold=0.25,
        device=device_str,
        label_to_class_id=DEART_CLASS_MAP,
    )

    t = target_label.strip().lower()
    candidates: List[Tuple[int, int, int, int, Image.Image, List[Tuple[int, int, int, int]], List[str]]] = []

    for i in range(min(scan_limit, len(ds))):
        img, _anns, meta = ds[i]
        boxes_raw, _scores_raw, labels_raw = grounded_sam._run_grounding_dino(img, DEART_TEXT_LABELS)
        labels_norm = [lbl.strip().lower().rstrip(".") for lbl in labels_raw]
        target_count = sum(lbl == t for lbl in labels_norm)
        if target_count <= 0:
            continue
        boxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes_raw]
        total_count = len(boxes)
        candidates.append((target_count, total_count, i, int(meta.get("index", i)), img.copy(), boxes, labels_norm))

    if not candidates:
        raise RuntimeError(f"No detections found for target label: {target_label}")

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    picked = candidates[:top_k]

    panel_w = 900
    resized_panels: List[Image.Image] = []
    caption_lines: List[str] = []
    for target_count, total_count, _i, img_idx, img, boxes, labels_norm in picked:
        boxed = _draw_dino_boxes(img, boxes, labels_norm, target_label=t)
        panel = boxed.resize((panel_w, int(boxed.height * panel_w / boxed.width)), Image.Resampling.LANCZOS)
        resized_panels.append(panel)
        caption_lines.append(f"img {img_idx} | {t} detections: {target_count} | total detections: {total_count}")

    header_h = 120
    caption_h = 70
    gap = 28
    body_h = sum(p.height for p in resized_panels) + gap * (len(resized_panels) - 1) + caption_h * len(resized_panels)
    canvas = Image.new("RGB", (panel_w + 40, header_h + body_h + 30), "white")
    d = ImageDraw.Draw(canvas)

    title_f = _font(40, bold=True)
    cap_f = _font(24)
    sub_f = _font(22)
    d.text((20, 24), f"GroundingDINO examples: {t}", fill="black", font=title_f)
    d.text((20, 76), "Red = target label boxes | Green = other predicted boxes", fill=(70, 70, 70), font=sub_f)

    y = header_h
    for panel, cap in zip(resized_panels, caption_lines):
        canvas.paste(panel, (20, y))
        y += panel.height
        d.text((20, y + 12), cap, fill=(60, 60, 60), font=cap_f)
        y += caption_h + gap

    out = OUT_DIR / f"dino_{t.replace(' ', '_')}_examples.png"
    canvas.save(out)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outs: List[Path] = []
    outs.append(make_metrics_summary())
    outs.append(make_mask_iou_explainer())
    outs.append(make_mask_iou_real_example())
    outs.append(make_text_prompt_pipeline_example())
    outs.append(make_gt_box_pipeline_example())
    outs.append(make_overdet_examples())
    outs.append(make_dino_class_examples("crucifixion", top_k=2, scan_limit=350))
    outs.append(make_dino_class_examples("shepherd", top_k=2, scan_limit=350))

    print("Generated slide assets:")
    for p in outs:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
