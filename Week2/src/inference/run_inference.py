import argparse
import time
import os
from pathlib import Path
from typing import Any, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
import wandb
from scipy.ndimage import label
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import KITTIMOTS
from models.sam_wrapper import SamWrapper
from prompting.grid import GridPromptStrategy
from prompting.sift import SiftPromptStrategy
import pycocotools.mask as rletools

def get_connected_components(mask: np.ndarray) -> int:
    """Calculates the number of connected components in a binary mask (fragmentation)."""
    labeled_array, num_features = label(mask)
    return num_features

def draw_prompts(image: Image.Image, prompt_data: dict) -> Image.Image:
    """Draws prompts (points, boxes) onto a copy of the image."""
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
            
    return vis

def draw_gt_masks(image: Image.Image, anns: list) -> Image.Image:
    """Draws Ground Truth masks on the image."""
    vis = image.copy()
    overlay = np.array(vis).copy()
    
    for ann in anns:
        mask = rletools.decode(ann.mask_rle).astype(np.uint8)
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        overlay[mask == 1] = color
        
    vis_overlay = Image.fromarray(
        (0.6 * np.array(vis) + 0.4 * overlay).astype(np.uint8)
    )
    return vis_overlay

def draw_pred_masks(image: Image.Image, masks_np: list) -> Image.Image:
    """Draws predicted masks on the image."""
    vis = image.copy()
    overlay = np.array(vis).copy()
    
    for mask in masks_np:
        if mask.max() == 0: continue
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        # Assuming boolean mask or 0/1 mask
        overlay[mask > 0] = color
        
    vis_overlay = Image.fromarray(
        (0.6 * np.array(vis) + 0.4 * overlay).astype(np.uint8)
    )
    return vis_overlay

def create_3pane_vertical(gt_img, prompt_img, pred_img):
    """Concatenates GT, Prompt, and Prediction images vertically with titles."""
    from PIL import Image, ImageDraw, ImageFont
    
    titles = ["Ground Truth", "Prompt", "Prediction"]
    images = [gt_img, prompt_img, pred_img]
    
    w, h = gt_img.size
    
    title_h = 40
    total_h = (h + title_h) * 3
    
    pane = Image.new("RGB", (w, total_h), "white")
    draw = ImageDraw.Draw(pane)
    
    # Font normal
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
        # using facebook/sam-vit-base by default if weights is none
        model_id = args.weights if args.weights else "facebook/sam-vit-base"
        return SamWrapper(model_id=model_id, device=args.device)
    # Add grounded_sam etc natively when configured
    else:
        raise ValueError(f"Unknown model: {args.model}")

def build_prompt_strategy(args: argparse.Namespace) -> Any:
    name = args.prompt.lower()
    if name == "grid":
        return GridPromptStrategy()
    elif name == "sift":
        return SiftPromptStrategy()
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
    limit: Optional[int] = None,
    log_tables: bool = False
) -> None:
    
    ds = KITTIMOTS(root=root, split=split, ann_source=ann_source, compute_boxes=True)
    
    run_name = f"{model_name}_{prompt_strategy}_inference"
    out_path = Path(output_dir) / run_name
    out_path.mkdir(parents=True, exist_ok=True)
    
    n = len(ds) if limit is None else min(limit, len(ds))
    
    print(f"Starting inference on {n} frames using model {type(model).__name__} and strategy {type(strategy).__name__}...")
    
    if torch.cuda.is_available():
         torch.cuda.reset_peak_memory_stats()
         
    total_inference_time = 0.0
    total_latency_start = time.time()
    
    for i in tqdm(range(n)):
        image, anns, meta = ds[i]
        
        # Generate Prompt
        prompt_data = strategy.generate_prompt(image, anns)
        
        # Predict
        masks_tensors, scores_tensors, inference_time = model.predict(image, prompt_data)
        total_inference_time += inference_time
        
        # Parse masks and scores. HuggingFace output for Batch=1
        masks_out = masks_tensors[0].squeeze(0) if len(masks_tensors[0].shape) > 3 else masks_tensors[0]
        scores_out = scores_tensors[0].squeeze(0) if len(scores_tensors[0].shape) > 2 else scores_tensors[0]
        
        # Extract the highest confidence mask for each prompt
        best_masks = []
        best_scores = []
        area_metrics = []
        frag_metrics = []
        
        if len(scores_out.shape) == 1: # Single prompt (shape [3])
            best_idx = torch.argmax(scores_out)
            b_mask = masks_out[best_idx].cpu().numpy()
            b_score = scores_out[best_idx].item()
            best_masks.append(b_mask)
            best_scores.append(b_score)
            
            area_metrics.append(np.sum(b_mask))
            frag_metrics.append(get_connected_components(b_mask))
        else:
            if len(masks_out.shape) == 3:
                if len(scores_out.shape) == 2:
                    scores_out = scores_out.squeeze(0)
                    
                best_idx = torch.argmax(scores_out)
                b_mask = masks_out[best_idx].cpu().numpy()
                b_score = scores_out[best_idx].item()
                best_masks.append(b_mask)
                best_scores.append(b_score)
                area_metrics.append(np.sum(b_mask))
                frag_metrics.append(get_connected_components(b_mask))
                
            else:
                for j in range(scores_out.shape[0]):
                    best_idx = torch.argmax(scores_out[j])
                    b_mask = masks_out[j, best_idx].cpu().numpy()
                    b_score = scores_out[j, best_idx].item()
                    best_masks.append(b_mask)
                    best_scores.append(b_score)
                    
                    area_metrics.append(np.sum(b_mask))
                    frag_metrics.append(get_connected_components(b_mask))
                
        # Visualization
        gt_img = draw_gt_masks(image, anns)
        prompt_img = draw_prompts(image, prompt_data)
        pred_img = draw_pred_masks(image, best_masks)
        
        pane_img = create_3pane_vertical(gt_img, prompt_img, pred_img)
        
        img_filename = f"{i:04d}_seq{meta['seq']}_frame{meta['frame']}.png"
        pane_img.save(out_path / img_filename)
        
        # Logging Iteration Metrics
        wandb.log({
            "inference/time_per_image": inference_time,
            "inference/num_prompts": len(best_masks),
            "inference/mask_count": len(best_masks),
            "inference/avg_confidence": np.mean(best_scores) if best_scores else 0,
            "inference/avg_mask_area": np.mean(area_metrics) if area_metrics else 0,
            "inference/avg_fragmentation": np.mean(frag_metrics) if frag_metrics else 0
        })
        
        if log_tables:
            table = wandb.Table(columns=["image_id", "mask_id", "confidence", "area", "fragmentation"])
            for m_id, (conf, area, frag) in enumerate(zip(best_scores, area_metrics, frag_metrics)):
                table.add_data(img_filename, m_id, conf, area, frag)
            wandb.log({f"mask_details_img_{i}": table})
            
    total_latency = time.time() - total_latency_start
    avg_time = total_inference_time / n
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    max_vram_mb = 0
    if torch.cuda.is_available():
        max_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
    wandb.log({
        "inference/avg_latency_ms": avg_time * 1000,
        "inference/fps": fps,
        "inference/total_time_s": total_inference_time,
        "inference/max_vram_mb": max_vram_mb,
        "inference/total_run_latency": total_latency
    })
    
    print(f"Finished. Total Latency: {total_latency:.2f}s")
    print(f"Saved predictions to {out_path}")
    

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on KITTI-MOTS dataset")

    # Dataset arguments (matching week 1)
    parser.add_argument("--root", type=str, default="~/mcv/datasets/C5/KITTI-MOTS/", help="Path to KITTI-MOTS dataset")
    parser.add_argument("--exp_name", type=str, default="default_experiment", help="Experiment name")
    parser.add_argument("--split", type=str, default="validation", help="dev or validation")
    parser.add_argument("--ann_source", type=str, default="txt", help="Annotation source (txt/png)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames for debugging")
    
    # Output argument modified to be output directory
    default_out_dir = str(Path(__file__).resolve().parent.parent / "results_inference")
    parser.add_argument("--output_dir", type=str, default=default_out_dir, help="Output directory for visualizations.")
    
    # Model arguments (matching week 1)
    parser.add_argument("--model", type=str, required=True, help="Model type: sam, grounded_sam")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights (HuggingFace ID for SAM)")
    parser.add_argument("--conf", type=float, default=0.0, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half", action="store_true", help="Use FP16")
    
    # Week 2 specific arguments
    parser.add_argument("--prompt", type=str, default="sift", choices=["grid", "sift", "bbox", "text"], help="Prompt strategy.")
    parser.add_argument("--log_tables", action="store_true", help="Log detailed tables of area vs confidence for every mask.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = build_model(args)
    strategy = build_prompt_strategy(args)
    
    # Extract underlying torch model param calculation
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

    # Inference
    run_inference(
        root=args.root,
        split=args.split,
        ann_source=args.ann_source,
        output_dir=args.output_dir,
        model_name=args.model,
        prompt_strategy=args.prompt,
        model=model,
        strategy=strategy,
        limit=args.limit,
        log_tables=args.log_tables
    )

    wandb.finish()
