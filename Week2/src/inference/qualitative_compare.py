"""
qualitative_compare.py

Side-by-side qualitative comparison between the pretrained SAM and a finetuned
SAM checkpoint.  For each sampled frame the script produces a 4-panel image:

    | Original | GT masks | Pretrained SAM | Finetuned SAM |

Usage (from Week2/):
    python -m src.qualitative_compare \
        --finetuned_weights results_finetune/sam_base_lh35g5yk/best_model.pth \
        --n_samples 20 \
        --prompt center_bb_gt \
        --output_dir results_qualitative
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import SamModel, SamProcessor
import pycocotools.mask as rletools

# ---------------------------------------------------------------------------
# Path bootstrap so we can import sibling packages
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import KITTIMOTS
from prompting.center_bb_gt import CenterBBGTPromptStrategy
from prompting.grid import GridPromptStrategy
from prompting.sift import SiftPromptStrategy


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_PALETTE = [
    (255, 50,  50),
    (50,  200, 50),
    (50,  100, 255),
    (255, 200, 0),
    (255, 0,   200),
    (0,   220, 220),
    (255, 128, 0),
    (160, 50,  255),
    (0,   255, 128),
    (255, 255, 100),
]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _overlay_masks(image: Image.Image, masks: List[np.ndarray], alpha: float = 0.55) -> Image.Image:
    """Blend coloured masks onto *image* and return a new PIL image."""
    base = np.array(image).copy()
    overlay = base.copy()
    for idx, mask in enumerate(masks):
        binary = mask > 0
        if not binary.any():
            continue
        color = np.array(_PALETTE[idx % len(_PALETTE)], dtype=np.uint8)
        overlay[binary] = color
    blended = (alpha * overlay + (1 - alpha) * base).astype(np.uint8)
    return Image.fromarray(blended)


def _overlay_gt(image: Image.Image, anns: list, alpha: float = 0.45) -> Image.Image:
    """Overlay ground-truth RLE masks on *image*."""
    base = np.array(image).copy()
    overlay = base.copy()
    rng = random.Random(0)
    for ann in anns:
        mask = rletools.decode(ann.mask_rle).astype(np.uint8)
        color = np.array([rng.randint(80, 255) for _ in range(3)], dtype=np.uint8)
        overlay[mask == 1] = color
    blended = (alpha * overlay + (1 - alpha) * base).astype(np.uint8)
    return Image.fromarray(blended)


def _add_title(img: Image.Image, title: str, bar_h: int = 38) -> Image.Image:
    """Paste a title bar above *img* and return the combined image."""
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    tx = (w - (bbox[2] - bbox[0])) // 2
    ty = (bar_h - (bbox[3] - bbox[1])) // 2
    draw.text((tx, ty), title, fill=(30, 30, 30), font=font)
    canvas.paste(img, (0, bar_h))
    return canvas


def make_comparison_strip(
    image: Image.Image,
    anns: list,
    pretrained_masks: List[np.ndarray],
    finetuned_masks: List[np.ndarray],
    meta: Dict[str, Any],
) -> Image.Image:
    """Create a vertical 4-panel stack with titles."""
    panels = [
        _add_title(image.copy(),              "Original"),
        _add_title(_overlay_gt(image, anns),  "Ground Truth"),
        _add_title(_overlay_masks(image, pretrained_masks), "Pretrained SAM"),
        _add_title(_overlay_masks(image, finetuned_masks),  "Finetuned SAM"),
    ]
    max_w   = max(p.width  for p in panels)
    total_h = sum(p.height for p in panels)
    strip = Image.new("RGB", (max_w, total_h), (240, 240, 240))
    y = 0
    for panel in panels:
        strip.paste(panel, (0, y))
        y += panel.height
    return strip


# ---------------------------------------------------------------------------
# SAM inference helper
# ---------------------------------------------------------------------------

class _SAMRunner:
    """Thin wrapper: load model, run forward pass, return best masks."""

    def __init__(
        self,
        model_id: str,
        weights_path: Optional[str],
        device: str,
        multimask_output: bool = True,
        use_box_prompts: bool = False,
    ):
        self.device = device
        self.multimask_output = multimask_output
        self.use_box_prompts = use_box_prompts
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device)

        if weights_path:
            ckpt = torch.load(weights_path, map_location=device, weights_only=False)
            # Support both raw state_dict and {"model_state_dict": ...} dicts
            state_dict = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  [warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            print(f"  Loaded finetuned weights from {weights_path}")

        self.model.eval()

    @torch.no_grad()
    def predict_masks(self, image: Image.Image, prompt_data: Dict[str, Any]) -> List[np.ndarray]:
        """Return binary masks — one per annotated object."""
        boxes = prompt_data.get("boxes")

        if self.use_box_prompts and boxes is not None and len(boxes) > 0:
            # Replicate exactly how prepare_batch_for_sam worked during training:
            # batched_input_boxes = [ [list_of_boxes] ]  (batch=1, all N boxes together)
            raw_boxes = [b.tolist() for b in boxes]
            batched_input_boxes = [raw_boxes]   # shape hint: [1, N, 4]

            img_np = np.array(image)
            inputs = self.processor(
                images=[img_np],
                input_boxes=[batched_input_boxes],   # [batch, [N boxes]]
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(self.device)
            input_boxes  = inputs["input_boxes"].to(self.device)

            outputs = self.model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=False,
            )

            # outputs.pred_masks: (1, 1, N, 1, 256, 256)  with multimask=False
            # reshape to (N, 256, 256) logits
            B = 1
            pred_masks = outputs.pred_masks.view(
                B, -1, outputs.pred_masks.shape[-2], outputs.pred_masks.shape[-1]
            )  # (1, N, 256, 256)
            pred_logits = pred_masks[0]  # (N, 256, 256)

            # Upscale to original image size (same as training eval)
            h, w = img_np.shape[:2]
            pred_upscaled = F.interpolate(
                pred_logits.unsqueeze(1),       # (N, 1, 256, 256)
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)                        # (N, h, w)

            pred_binary = (torch.sigmoid(pred_upscaled) > 0.5).cpu().numpy()  # (N, h, w) bool
            return [pred_binary[j] for j in range(pred_binary.shape[0])]

        # ---- Pretrained path: use point/box prompts with post_process_masks ----
        p_type = prompt_data.get("type")
        if p_type in ("point", "point_and_box"):
            points_list = prompt_data["points"].tolist()
            labels_list = prompt_data["point_labels"].tolist()
            input_points = [[[pt] for pt in points_list]]
            input_labels = [[[lb] for lb in labels_list]]
            inputs = self.processor(
                image, input_points=input_points, input_labels=input_labels,
                return_tensors="pt"
            ).to(self.device)
        elif p_type == "box" and boxes is not None:
            input_boxes_fmt = [[[box.tolist()] for box in boxes]]
            inputs = self.processor(image, input_boxes=input_boxes_fmt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs, multimask_output=self.multimask_output)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        scores = outputs.iou_scores.cpu()
        masks_out  = masks[0]    # (N, n_candidates, H, W)
        scores_out = scores[0]   # (N, n_candidates)
        best_masks = []
        for j in range(scores_out.shape[0]):
            best_idx = torch.argmax(scores_out[j]).item()
            best_masks.append(masks_out[j, best_idx].numpy().astype(bool))
        return best_masks


# ---------------------------------------------------------------------------
# Prompt strategy factory
# ---------------------------------------------------------------------------

def _build_strategy(name: str):
    name = name.lower()
    if name == "center_bb_gt":
        return CenterBBGTPromptStrategy()
    elif name == "grid":
        return GridPromptStrategy()
    elif name == "sift":
        return SiftPromptStrategy()
    raise ValueError(f"Unknown prompt strategy: {name!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qualitative comparison: pretrained vs finetuned SAM")
    p.add_argument("--root", default="~/mcv/datasets/C5/KITTI-MOTS/",
                   help="Path to KITTI-MOTS dataset root")
    p.add_argument("--split", default="validation", choices=["train", "dev", "validation"],
                   help="Dataset split to sample from")
    p.add_argument("--ann_source", default="txt", choices=["txt", "png"])
    p.add_argument("--model_id", default="facebook/sam-vit-base",
                   help="HuggingFace model ID for the base SAM (used for both models)")
    p.add_argument("--finetuned_weights", required=True,
                   help="Path to the finetuned model checkpoint (.pth)")
    p.add_argument("--prompt", default="center_bb_gt",
                   choices=["center_bb_gt", "grid", "sift"],
                   help="Prompt strategy to use for both models")
    p.add_argument("--n_samples", type=int, default=20,
                   help="Number of frames to visualise")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for frame sampling")
    p.add_argument("--output_dir", default="results_qualitative",
                   help="Directory to save comparison images")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_empty", action="store_true",
                   help="Skip frames with no annotations")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset ----
    ds = KITTIMOTS(root=args.root, split=args.split, ann_source=args.ann_source, compute_boxes=True)
    print(f"Dataset split='{args.split}' — {len(ds)} frames")

    # Sample indices
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    strategy = _build_strategy(args.prompt)

    # Filter to frames that actually have annotations when --skip_empty
    selected: List[int] = []
    for idx in indices:
        if len(selected) >= args.n_samples:
            break
        _, anns, _ = ds[idx]
        if args.skip_empty and len(anns) == 0:
            continue
        prompt_data = strategy.generate_prompt(ds[idx][0], anns)
        n_prompts = (
            len(prompt_data.get("points", [])) if prompt_data.get("type") in ("point", "point_and_box")
            else len(prompt_data.get("boxes", []))
        )
        if n_prompts == 0:
            continue
        selected.append(idx)

    print(f"Selected {len(selected)} frames for comparison")

    # ---- Models ----
    print("\nLoading pretrained SAM …")
    pretrained = _SAMRunner(
        args.model_id, weights_path=None, device=args.device,
        multimask_output=True, use_box_prompts=False,
    )
    print("Loading finetuned SAM …")
    # The finetuned model was trained with multimask_output=False and box prompts.
    finetuned = _SAMRunner(
        args.model_id, weights_path=args.finetuned_weights, device=args.device,
        multimask_output=False, use_box_prompts=True,
    )

    # ---- Inference + Visualisation ----
    print(f"\nRunning inference on {len(selected)} frames …")
    for rank, idx in enumerate(tqdm(selected, desc="Comparing")):
        image, anns, meta = ds[idx]
        prompt_data = strategy.generate_prompt(image, anns)

        pretrained_masks = pretrained.predict_masks(image, prompt_data)
        finetuned_masks  = finetuned.predict_masks(image, prompt_data)

        strip = make_comparison_strip(image, anns, pretrained_masks, finetuned_masks, meta)

        seq   = meta.get("seq",   "?")
        frame = meta.get("frame", idx)
        fname = f"{rank:03d}_seq{seq}_frame{frame:06d}.jpg"
        strip.save(out_dir / fname, quality=92)

    print(f"\nSaved {len(selected)} comparison images to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
