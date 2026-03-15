#!/usr/bin/env python
"""
Batch evaluation of all finetuned SAM models in a given directory.

For each model sub-directory, loads the best checkpoint and runs
segmentation evaluation using the prompt type inferred from the directory
name (bbox / point / text / mix). Results are saved to a single JSON file.

Usage:
    python -m src.inference.batch_eval_finetuned \
        --finetuned_dir results_finetune/final_finetuned \
        --split         validation \
        --output        results_eval/finetuned_comparison_metrics.json
"""

import os
import json
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor, AutoProcessor, AutoModelForZeroShotObjectDetection

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from Week2.src.datasets import KITTIMOTS
from Week2.src.finetune.sam_finetune import collate_fn
from Week2.src.inference.evaluation_segm import CocoSegmentationMetrics
from Week2.src.finetune.eval_sam_metrics import run_single_pass


def _detect_prompt_type(dir_name: str, default: str) -> str:
    for pt in ("bbox", "point", "text", "mix"):
        if pt in dir_name:
            return pt
    return default


def batch_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = KITTIMOTS(
        root=args.root, split=args.split, ann_source="txt",
        seed=42, split_ratio=0.8, compute_boxes=True,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    coco_metrics = CocoSegmentationMetrics(
        root=args.root, dataset_name="kitti_mots", split=args.split,
        ann_source="txt", seed=42, split_ratio=0.8,
    )

    dino_model, dino_processor = None, None
    dino_id = "IDEA-Research/grounding-dino-tiny"

    finetune_root = Path(args.finetuned_dir)
    model_dirs = sorted(d for d in finetune_root.iterdir() if d.is_dir())
    print(f"Found {len(model_dirs)} model(s) in {finetune_root}")

    results_summary = {}

    for model_dir in model_dirs:
        print(f"\nEvaluating: {model_dir.name}")

        weight_path = model_dir / "best_model.pth"
        if not weight_path.exists():
            print(f"  [SKIP] No best_model.pth found.")
            continue

        prompt_type = _detect_prompt_type(model_dir.name, args.prompt_type)

        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
        print(f"  Loaded weights from {weight_path}")

        if prompt_type in ("text", "mix") and dino_model is None:
            print(f"  Loading GroundingDINO ({dino_id})...")
            dino_processor = AutoProcessor.from_pretrained(dino_id)
            dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)
            dino_model.eval()

        prompt_types_to_eval = ["bbox", "point", "text"] if prompt_type == "mix" else [prompt_type]
        model_results = {}
        for p_type in prompt_types_to_eval:
            print(f"  -> prompt_type={p_type}")
            model_results[p_type] = run_single_pass(
                loader, processor, model, device,
                p_type, args.text_prompt, coco_metrics,
                dino_model, dino_processor,
            )

        results_summary[model_dir.name] = model_results

    output_file = Path(args.output).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results_summary, f, indent=4)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluate all finetuned SAM models.")
    parser.add_argument("--root",         default="/ghome/group01/mcv/datasets/C5/KITTI-MOTS/")
    parser.add_argument("--finetuned_dir",default="/ghome/group01/C5/benet/C5-Team1/Week2/results_finetune/final_finetuned")
    parser.add_argument("--split",        default="validation")
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--prompt_type",  default="bbox",
                        help="Fallback prompt type if not detected from directory name.")
    parser.add_argument("--text_prompt",  default="Person. Car.")
    parser.add_argument("--output",       default="results_eval/finetuned_comparison_metrics.json")
    batch_evaluate(parser.parse_args())
