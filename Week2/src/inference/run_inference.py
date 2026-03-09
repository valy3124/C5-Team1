import argparse
import time
import os
from pathlib import Path
from typing import Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import wandb
from scipy.ndimage import label
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import KITTIMOTS
from models.sam_wrapper import SamWrapper
from models.grounded_sam import GroundedSamWrapper
from prompting.grid import GridPromptStrategy
from prompting.sift import SiftPromptStrategy
from prompting.center_bb_gt import CenterBBGTPromptStrategy
from prompting.text import TextPromptStrategy
import pycocotools.mask as rletools

def get_connected_components(mask: np.ndarray) -> int:
    """Calculates the number of connected components in a binary mask (fragmentation)."""
    labeled_array, num_features = label(mask.astype(int))
    return num_features

def get_prompt_count(prompt_data: dict) -> int:
    """Safely extracts the number of prompts generated."""
    p_type = prompt_data.get("type")
    if p_type in ["point", "point_and_box"]:
        return len(prompt_data.get("points", []))
    elif p_type == "box":
        return len(prompt_data.get("boxes", []))
    return 1

def mask_nms(masks: list, scores: list, iou_threshold: float = 0.8):
    """
    Applies Non-Maximum Suppression (NMS) to predicted masks based on IoU.
    Discards overlapping masks, keeping the one with the highest confidence score.
    """
    if not masks:
        return [], []

    indices = np.argsort(scores)[::-1]
    
    keep_masks = []
    keep_scores = []
    
    for idx in indices:
        current_mask = masks[idx] > 0
        current_score = scores[idx]
        
        is_duplicate = False
        for kept_mask in keep_masks:
            intersection = np.logical_and(current_mask, kept_mask).sum()
            union = np.logical_or(current_mask, kept_mask).sum()
            
            iou = intersection / union if union > 0 else 0
            
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep_masks.append(current_mask)
            keep_scores.append(current_score)
            
    return keep_masks, keep_scores

def draw_prompts(image: Image.Image, prompt_data: dict) -> Image.Image:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    
    p_type = prompt_data.get("type")
    if p_type == "point":
        points = prompt_data.get("points", [])
        for pt in points:
            x, y = pt[0], pt[1]
            r = 3
            draw.ellipse((x - r, y - r, x + r, y + r), fill="red", outline="white")
    elif p_type == "box":
        boxes = prompt_data.get("boxes", [])
        for box in boxes:
            draw.rectangle(tuple(box), outline="red", width=3)
    elif p_type == "point_and_box":
        points = prompt_data.get("points", [])
        for pt in points:
            x, y = pt[0], pt[1]
            r = 3
            draw.ellipse((x - r, y - r, x + r, y + r), fill="red", outline="white")
            
        boxes = prompt_data.get("boxes", [])
        for box in boxes:
            draw.rectangle(tuple(box), outline="blue", width=3)
            
    return vis

def draw_gt_masks(image: Image.Image, anns: list) -> Image.Image:
    vis = image.copy()
    overlay = np.array(vis).copy()
    
    for idx, ann in enumerate(anns):
        mask = rletools.decode(ann.mask_rle).astype(np.uint8)
        color = np.array(_MASK_PALETTE[idx % len(_MASK_PALETTE)], dtype=np.uint8)
        overlay[mask == 1] = color
        
    vis_overlay = Image.fromarray(
        (0.6 * np.array(vis) + 0.4 * overlay).astype(np.uint8)
    )
    return vis_overlay

_MASK_PALETTE = [
    (255, 50,  50),   # red
    (50,  200, 50),   # green
    (50,  100, 255),  # blue
    (255, 200, 0),    # yellow
    (255, 0,   200),  # magenta
    (0,   220, 220),  # cyan
    (255, 128, 0),    # orange
    (160, 50,  255),  # purple
    (0,   255, 128),  # mint
    (255, 255, 100),  # lime yellow
]

def draw_pred_masks(image: Image.Image, masks_np: list, scores_np: list) -> Image.Image:
    """
    Draws predicted masks using strict pixel assignment.
    Prioritizes high-confidence masks so lower-confidence masks don't overwrite them.
    """
    vis = image.copy()
    overlay = np.array(vis).copy()
    
    if not masks_np:
        return vis

    sorted_indices = np.argsort(scores_np)[::-1]
    assigned_pixels = np.zeros(overlay.shape[:2], dtype=bool)
    
    for color_idx, mask_idx in enumerate(sorted_indices):
        mask = masks_np[mask_idx] > 0
        valid_mask = np.logical_and(mask, ~assigned_pixels)
        
        if not valid_mask.any():
            continue
            
        color = np.array(_MASK_PALETTE[color_idx % len(_MASK_PALETTE)], dtype=np.uint8)
        overlay[valid_mask] = color
        assigned_pixels = np.logical_or(assigned_pixels, valid_mask)
        
    vis_overlay = Image.fromarray(
        (0.45 * np.array(vis) + 0.55 * overlay).astype(np.uint8)
    )
    return vis_overlay

def create_3pane_vertical(gt_img: Image.Image, prompt_img: Image.Image, pred_img: Image.Image) -> Image.Image:
    titles = ["Ground Truth", "Prompt", "Prediction"]
    images = [gt_img, prompt_img, pred_img]
    
    w, h = gt_img.size
    title_h = 40
    total_h = (h + title_h) * 3
    
    pane = Image.new("RGB", (w, total_h), "white")
    draw = ImageDraw.Draw(pane)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except:
        font = ImageFont.load_default()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        y_offset = i * (h + title_h)
        
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        text_x = (w - text_w) // 2
        text_y = y_offset + (title_h - text_h) // 2
        
        draw.text((text_x, text_y), title, fill="black", font=font)
        pane.paste(img, (0, y_offset + title_h))
    
    return pane

def build_model(args: argparse.Namespace) -> Any:
    name = args.model.lower()
    if name == "sam":
        model_id = args.weights if args.weights else "facebook/sam-vit-base"
        return SamWrapper(model_id=model_id, device=args.device)
    elif name == "grounded_sam":
        dino_id = "IDEA-Research/grounding-dino-tiny"
        sam_id  = "facebook/sam-vit-base"
        if args.weights:
            parts = args.weights.split("|")
            dino_id = parts[0].strip()
            if len(parts) > 1:
                sam_id = parts[1].strip()
        return GroundedSamWrapper(
            dino_model_id=dino_id,
            sam_model_id=sam_id,
            box_threshold=args.conf if args.conf > 0 else 0.70,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

def build_prompt_strategy(args: argparse.Namespace) -> Any:
    name = args.prompt.lower()
    if name == "grid":
        return GridPromptStrategy()
    elif name == "sift":
        return SiftPromptStrategy()
    elif name == "center_bb_gt":
        return CenterBBGTPromptStrategy()
    elif name == "text":
        return TextPromptStrategy(text_labels=args.text_labels)
    else:
        raise ValueError(f"Unknown prompt strategy: {args.prompt}")

def run_inference(
    root: str,
    split: str,
    ann_source: str,
    output_dir: str,
    model_name: str,
    prompt_strategy: str,
    model: Any,
    strategy: Any,
    exp_name: str = "default_experiment",
    limit: Optional[int] = None
) -> None:
    
    ds = KITTIMOTS(root=root, split=split, ann_source=ann_source, compute_boxes=True)
    
    run_name = exp_name if exp_name != "default_experiment" else f"{model_name}_{prompt_strategy}_inference"
    out_path = Path(output_dir) / run_name
    out_path.mkdir(parents=True, exist_ok=True)
    
    n = len(ds) if limit is None else min(limit, len(ds))
    print(f"Starting inference on {n} frames using model {type(model).__name__} and strategy {type(strategy).__name__}...")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
         
    total_inference_time = 0.0
    total_latency_start = time.time()
    
    global_image_ids = []
    global_num_bbs = []
    global_confidences = []
    global_areas = []
    global_fragmentations = []
    total_prompts_sent = 0
    total_masks_predicted = 0
    
    with torch.no_grad():
        for i in tqdm(range(n), desc="Processing Frames"):
            image, anns, meta = ds[i]
            
            prompt_data = strategy.generate_prompt(image, anns)
            num_prompts = get_prompt_count(prompt_data)
            
            if num_prompts == 0:
                continue
                
            total_prompts_sent += num_prompts
            
            masks_tensors, scores_tensors, inference_time = model.predict(image, prompt_data)
            total_inference_time += inference_time
            
            masks_out = masks_tensors[0]
            if masks_out.dim() == 5:
                masks_out = masks_out.squeeze(0)
            scores_out = scores_tensors 
            if scores_out.dim() == 3:
                scores_out = scores_out.squeeze(0)
            
            raw_masks = []
            raw_scores = []
            
            n_objects = scores_out.shape[0]
            for j in range(n_objects):
                best_idx = torch.argmax(scores_out[j])
                b_mask  = masks_out[j, best_idx].cpu().numpy()
                b_score = scores_out[j, best_idx].item()
                
                raw_masks.append(b_mask)
                raw_scores.append(b_score)
                
            best_masks, best_scores = mask_nms(raw_masks, raw_scores, iou_threshold=0.8)
            
            for b_mask, b_score in zip(best_masks, best_scores):
                global_image_ids.append(meta['image_path'].split('/')[-1])
                global_num_bbs.append(num_prompts)
                global_confidences.append(b_score)
                
                binary_mask = b_mask > 0
                n_components = get_connected_components(binary_mask)
                global_areas.append(np.sum(binary_mask))
                global_fragmentations.append(n_components)
            
            total_masks_predicted += len(best_masks)
            
            if i % 20 == 0:
                gt_img = draw_gt_masks(image, anns)
                prompt_img = draw_prompts(image, prompt_data)
                
                pred_img = draw_pred_masks(image, best_masks, best_scores)
                
                pane_img = create_3pane_vertical(gt_img, prompt_img, pred_img)
                img_filename = f"{i:04d}_seq{meta['seq']}_frame{meta['frame']}.png"
                pane_img.save(out_path / img_filename)
            
    total_latency = time.time() - total_latency_start
    avg_time = total_inference_time / n if n > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    max_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    
    wandb.log({
        "metrics/avg_confidence": np.mean(global_confidences) if global_confidences else 0,
        "metrics/avg_mask_area_px": np.mean(global_areas) if global_areas else 0,
        "metrics/avg_fragmentation": np.mean(global_fragmentations) if global_fragmentations else 0,
        
        "metrics/total_prompts_sent": total_prompts_sent,
        "metrics/total_masks_predicted": total_masks_predicted,
        
        "performance/avg_latency_ms": avg_time * 1000,
        "performance/fps": fps,
        "performance/total_inference_time_s": total_inference_time,
        "performance/total_run_latency_s": total_latency,
        "performance/max_vram_mb": max_vram_mb
    })
    
    csv_path = out_path / "mask_metrics.csv"
    df_metrics = pd.DataFrame({
        "image_id": global_image_ids,
        "num_bb": global_num_bbs,
        "confidence": global_confidences,
        "mask_area": global_areas,
        "num_fragments": global_fragmentations,
        "is_fragmented": [int(f > 1) for f in global_fragmentations]
    })
    df_metrics.to_csv(csv_path, index=False)
    
    print(f"Finished. Total Latency: {total_latency:.2f}s | FPS: {fps:.2f}")
    print(f"Saved predictions to {out_path}")
    print(f"Saved mask metrics to {csv_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on KITTI-MOTS dataset")

    parser.add_argument("--root", type=str, default="~/mcv/datasets/C5/KITTI-MOTS/", help="Path to KITTI-MOTS dataset")
    parser.add_argument("--exp_name", type=str, default="default_experiment", help="Experiment name")
    parser.add_argument("--split", type=str, default="validation", help="dev or validation")
    parser.add_argument("--ann_source", type=str, default="txt", help="Annotation source (txt/png)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames for debugging")
    
    default_out_dir = str(Path(__file__).resolve().parent.parent / "results_inference")
    parser.add_argument("--output_dir", type=str, default=default_out_dir, help="Output directory for visualizations.")
    
    parser.add_argument("--model", type=str, required=True, help="Model type: sam, grounded_sam")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights (HuggingFace ID for SAM)")
    parser.add_argument("--conf", type=float, default=0.0, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half", action="store_true", help="Use FP16")
    
    parser.add_argument("--prompt", type=str, default="sift", choices=["grid", "sift", "bbox", "text", "center_bb_gt"], help="Prompt strategy.")
    parser.add_argument("--text_labels", type=str, default="pedestrian. car.", help="Period-separated class labels for text prompt strategy (Grounded SAM).")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = build_model(args)
    strategy = build_prompt_strategy(args)
    
    if hasattr(model, "model"):
        torch_model = model.model
    else:
        torch_model = model

    num_params = sum(p.numel() for p in torch_model.parameters()) if hasattr(torch_model, "parameters") else 0
    num_trainable = sum(p.numel() for p in torch_model.parameters() if p.requires_grad) if hasattr(torch_model, "parameters") else 0

    run_exp_name = f"{args.model}_{args.prompt}_inference"
    if args.exp_name != "default_experiment":
        run_exp_name = args.exp_name + "_" + run_exp_name
        
    wandb.init(
        entity="c5-team1",
        project="Task a) Project 2",
        name=run_exp_name
    )

    wandb.config.update({
        "model_name": args.model,
        "weights": args.weights,
        "prompt_strategy": args.prompt,
        "num_parameters": num_params,
        "num_trainable_parameters": num_trainable,
    })

    run_inference(
        root=args.root,
        split=args.split,
        ann_source=args.ann_source,
        output_dir=args.output_dir,
        model_name=args.model,
        prompt_strategy=args.prompt,
        model=model,
        strategy=strategy,
        exp_name=run_exp_name,
        limit=args.limit
    )

    wandb.finish()