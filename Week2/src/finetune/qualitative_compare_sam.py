#!/usr/bin/env python
"""
qualitative_compare_sam.py

Side-by-side qualitative comparison between the pretrained SAM and a finetuned
SAM checkpoint.  For each sampled frame the script produces a 4-panel image:

    | Original with GT BBoxes | GT masks | Pretrained SAM | Finetuned SAM |
"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import random

import numpy as np
import pycocotools.mask as rletools
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from Week2.src.datasets import KITTIMOTS, InstanceAnn

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

def _overlay_masks(image: Image.Image, masks: List[np.ndarray], alpha: float = 0.55) -> Image.Image:
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
    base = np.array(image).copy()
    overlay = base.copy()
    rng = random.Random(0)
    for ann in anns:
        mask = rletools.decode(ann.mask_rle).astype(np.uint8)
        color = np.array([rng.randint(80, 255) for _ in range(3)], dtype=np.uint8)
        overlay[mask == 1] = color
    blended = (alpha * overlay + (1 - alpha) * base).astype(np.uint8)
    return Image.fromarray(blended)

def _draw_bboxes(image: Image.Image, anns: list, prompt_type: str = "bbox") -> Image.Image:
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    for ann in anns:
        x1, y1, x2, y2 = ann.bbox_xyxy
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        if prompt_type == "point":
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill="lime")
            draw.text((cx + 6, cy - 10), "x", fill="lime")
    return img_copy

def _add_title(img: Image.Image, title: str, bar_h: int = 38) -> Image.Image:
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
    prompt_type: str = "bbox",
    text_prompt: str = "pedestrian. car.",
    dino_boxes: List[List[float]] = None
) -> Image.Image:
    if prompt_type == "point":
        title = "Original + GT BBoxes (Prompt: Points)"
    elif prompt_type == "text":
        title = "Original + GT BBoxes"
    else:
        title = "Original + GT BBoxes (Prompt: BBoxes)"
        
    panels = [
        _add_title(_draw_bboxes(image, anns, prompt_type), title),
        _add_title(_overlay_gt(image, anns),  "Ground Truth Masks"),
    ]
    
    if prompt_type == "text" and dino_boxes is not None:
        fake_anns = [InstanceAnn(object_id=0, class_id=1, instance_id=0, mask_rle={}, bbox_xyxy=b) for b in dino_boxes]
        panel_title = f"Input BBoxes from GroundedSAM (Prompt: Text '{text_prompt}')"
        panels.append(_add_title(_draw_bboxes(image, fake_anns, "bbox"), panel_title))
    
    panels.extend([
        _add_title(_overlay_masks(image, pretrained_masks), "Pretrained SAM"),
        _add_title(_overlay_masks(image, finetuned_masks),  "Finetuned SAM"),
    ])
    max_w   = max(p.width  for p in panels)
    total_h = sum(p.height for p in panels)
    strip = Image.new("RGB", (max_w, total_h), (240, 240, 240))
    y = 0
    for panel in panels:
        strip.paste(panel, (0, y))
        y += panel.height
    return strip

class _SAMRunner:
    def __init__(self, model_id: str, weights_path: Optional[str], device: str):
        self.device = device
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device)

        if weights_path:
            ckpt = torch.load(weights_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {weights_path} (missing={len(missing)})")
        
        self.model.eval()

    @torch.no_grad()
    def predict_masks(self, image: Image.Image, anns: list, prompt_type: str = "bbox", dino_model=None, dino_processor=None, text_prompt="pedestrian. car.") -> Tuple[List[np.ndarray], List[List[float]]]:
        img_np = np.array(image)
        
        if prompt_type == "text" and dino_model is not None:
            text_labels = text_prompt
            try:
                dino_inputs = dino_processor(images=image, text=[[lbl.strip() for lbl in text_labels.rstrip(".").split(".") if lbl.strip()]], return_tensors="pt").to(self.device)
            except Exception:
                dino_inputs = dino_processor(images=image, text=text_labels, return_tensors="pt").to(self.device)
            dino_outputs = dino_model(**dino_inputs)
            dino_results = dino_processor.post_process_grounded_object_detection(
                dino_outputs, dino_inputs.input_ids, threshold=0.35, text_threshold=0.25, target_sizes=[image.size[::-1]]
            )[0]
            boxes = dino_results["boxes"].cpu().tolist()
        else:
            boxes = [ann.bbox_xyxy for ann in anns]
            
        if not boxes:
            return [], boxes
            
        if prompt_type in ("bbox", "text"):
            raw_boxes = [list(b) for b in boxes]
            batched_input_boxes = [raw_boxes]
            inputs = self.processor(
                images=[img_np],
                input_boxes=[batched_input_boxes],
                return_tensors="pt",
            )
        elif prompt_type == "point":
            raw_points = []
            raw_labels = []
            for b in boxes:
                x1, y1, x2, y2 = b
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                raw_points.append([[cx, cy]])
                raw_labels.append([1])
                
            batched_input_points = [raw_points]
            batched_input_labels = [raw_labels]
            
            inputs = self.processor(
                images=[img_np],
                input_points=batched_input_points,
                input_labels=batched_input_labels,
                return_tensors="pt",
            )
            inputs["input_points"] = inputs["input_points"].view(1, len(boxes), 1, 2)
            inputs["input_labels"] = inputs["input_labels"].view(1, len(boxes), 1)
        else:
            raise ValueError(f"Unknown prompt type {prompt_type}")
        
        sam_kwargs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
            outputs = self.model(
                **sam_kwargs,
                multimask_output=False,
            )
            
            B = 1
            pred_masks = outputs.pred_masks.view(B, -1, outputs.pred_masks.shape[-2], outputs.pred_masks.shape[-1])
            pred_logits = pred_masks[0]
            
            orig_h, orig_w = inputs["original_sizes"][0].tolist()
            reshaped_h, reshaped_w = inputs["reshaped_input_sizes"][0].tolist()
            
            valid_preds = pred_logits.unsqueeze(1)
            
            up_masks = F.interpolate(valid_preds, size=(1024, 1024), mode="bilinear", align_corners=False)
            up_masks = up_masks[..., :reshaped_h, :reshaped_w]
            up_masks = F.interpolate(up_masks, size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(1)
            
            pred_binary = (torch.sigmoid(up_masks) > 0.5).cpu().numpy().astype(bool)
            
            return [pred_binary[j] for j in range(pred_binary.shape[0])], boxes

def parse_args():
    p = argparse.ArgumentParser(description="SAM qualitative comparison (Equally spaced samples)")
    p.add_argument("--root", default="/ghome/group01/mcv/datasets/C5/KITTI-MOTS/")
    p.add_argument("--split", default="validation", choices=["dev", "validation"])
    p.add_argument("--model_id", default="facebook/sam-vit-base")
    p.add_argument("--finetuned_dir", required=True, help="Path to finetuned model directory containing best_model.pth")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--output_dir", default="results_qualitative_sam")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt_type", default="bbox", choices=["bbox", "point", "text"])
    p.add_argument("--text_prompt", default="pedestrian. car.", type=str, help="Text labels for GroundingDINO when prompt_type is text")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    finetuned_weights = Path(args.finetuned_dir) / "best_model.pth"
    if not finetuned_weights.exists():
        print(f"Error: {finetuned_weights} does not exist", file=sys.stderr)
        sys.exit(1)
        
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ds = KITTIMOTS(root=args.root, split=args.split, ann_source="txt", seed=args.seed, compute_boxes=True)
    print(f"Dataset split='{args.split}' - {len(ds)} frames")
    
    valid_indices = []
    for idx in range(len(ds)):
        _, anns, _ = ds[idx]
        if len(anns) > 0:
            valid_indices.append(idx)
            
    if not valid_indices:
        print("No frames with annotations found.")
        return
        
    n_samples = min(args.n_samples, len(valid_indices))
    step = len(valid_indices) / n_samples
    selected = [valid_indices[int(i * step)] for i in range(n_samples)]
    
    print(f"Selected {len(selected)} equally spaced frames with annotations: {selected}")
    
    dino_model = None
    dino_processor = None
    if args.prompt_type == "text":
        dino_id = "IDEA-Research/grounding-dino-tiny"
        print(f"\nLoading GroundingDINO {dino_id} ...")
        dino_processor = AutoProcessor.from_pretrained(dino_id)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)
        dino_model.eval()
    
    print("\nLoading models ...")
    pretrained = _SAMRunner(args.model_id, weights_path=None, device=device)
    finetuned = _SAMRunner(args.model_id, weights_path=str(finetuned_weights), device=device)
    
    print(f"\nRunning inference on {len(selected)} frames ...")
    for _, idx in enumerate(tqdm(selected, desc="Comparing (Equally Spaced)")):
        image, anns, meta = ds[idx]
        
        pretrained_masks, dino_boxes = pretrained.predict_masks(image, anns, args.prompt_type, dino_model, dino_processor, args.text_prompt)
        finetuned_masks, _  = finetuned.predict_masks(image, anns, args.prompt_type, dino_model, dino_processor, args.text_prompt)
        
        strip = make_comparison_strip(image, anns, pretrained_masks, finetuned_masks, meta, args.prompt_type, args.text_prompt, dino_boxes)
        
        seq   = meta.get("seq",   "?")
        frame = meta.get("frame", idx)
        fname = f"seq{seq}_frame{frame:06d}.jpg"
        strip.save(out_dir / fname, quality=92)
        
    print(f"\nSaved {len(selected)} comparison images to: {out_dir}")

if __name__ == "__main__":
    main()
