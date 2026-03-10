#!/usr/bin/env python
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SamModel, SamProcessor

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from Week2.src.datasets import KITTIMOTS
import pycocotools.mask as rletools
from Week2.src.finetune.sam_finetune import (
    DiceBCELoss, 
    collate_fn, 
    prepare_batch_for_sam, 
    postprocess_preds_and_flatten
)
from Week2.src.inference.evaluation_segm import CocoSegmentationMetrics

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model ID: {args.model_id} | Split: {args.split}")

    ds = KITTIMOTS(
        root=args.root, split=args.split, ann_source="txt",
        seed=args.seed, split_ratio=args.split_ratio, compute_boxes=True,
    )
    print(f"{args.split} split: {len(ds)} frames")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    print(f"Loading {args.model_id} ...")
    processor = SamProcessor.from_pretrained(args.model_id)
    model = SamModel.from_pretrained(args.model_id).to(device)
    model.eval()

    loss_fn = DiceBCELoss()

    coco_metrics = CocoSegmentationMetrics(
        root=args.root,
        dataset_name="kitti_mots",
        split=args.split,
        ann_source="txt",
        seed=args.seed,
        split_ratio=args.split_ratio
    )
    
    total_loss = total_bce = total_dice = 0.0
    coco_dt_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating SAM on {args.split}"):
            sam_kwargs, raw_masks, num_boxes, valid_metas_list, valid_targets_list, original_sizes, reshaped_input_sizes = \
                prepare_batch_for_sam(batch, processor, device, args.prompt_type)
            if sam_kwargs is None:
                continue

            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(
                    **sam_kwargs,
                    multimask_output=False
                )

                pred_1d, gt_1d, pred_list, real_ious_1d, pred_ious_1d = postprocess_preds_and_flatten(
                    outputs, raw_masks, num_boxes, original_sizes, reshaped_input_sizes, device
                )

                loss_seg, bce, dice = loss_fn(pred_1d, gt_1d)
                loss_iou = F.mse_loss(pred_ious_1d, real_ious_1d)
                loss = loss_seg + loss_iou

            total_loss += loss.item(); total_bce += bce.item(); total_dice += dice.item()

            iou_scores_out = outputs.iou_scores.view(len(num_boxes), -1).cpu()
            for i, (n, tgt, meta) in enumerate(zip(num_boxes, valid_targets_list, valid_metas_list)):
                if n == 0 or pred_list[i] is None:
                    continue
                image_id = meta["index"]
                
                mask_logits_i_resized = pred_list[i].cpu()
                iou_scores_i = iou_scores_out[i, :n].numpy()
                
                pred_binary = (torch.sigmoid(mask_logits_i_resized) > 0.5).numpy().astype(np.uint8)
                scores = iou_scores_i
                
                safe_n = min(n, len(tgt))
                for j in range(safe_n):
                    cat_id = tgt[j].class_id
                    if cat_id not in coco_metrics.label_map: continue
                    
                    coco_cat_id = coco_metrics.label_map[cat_id]
                    mask_j = np.asfortranarray(pred_binary[j])
                    rle = rletools.encode(mask_j)
                    rle["counts"] = rle["counts"].decode("utf-8")
                    bbox = rletools.toBbox(rle).tolist()
                    
                    coco_dt_list.append({
                        "image_id": image_id,
                        "category_id": coco_cat_id,
                        "segmentation": rle, "bbox": bbox,
                        "score": float(scores[j]),
                    })

    avg_loss = total_loss / max(1, len(loader))
    avg_bce  = total_bce  / max(1, len(loader))
    avg_dice = total_dice / max(1, len(loader))
    
    metrics = {"loss": avg_loss, "loss_bce": avg_bce, "loss_dice": avg_dice}

    if len(coco_dt_list) > 0:
        coco_dt = coco_metrics.coco_gt.loadRes(coco_dt_list)
        coco_res = coco_metrics.compute_metrics(coco_dt)
        metrics.update(coco_res)
        metrics["dice"] = metrics.get("overall/AP_segm", 0.0)
    else:
        metrics["dice"] = avg_loss

    print(f"\n=== SAM metrics | split={args.split} | pretrained weights ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate pretrained SAM on dev or validation split")
    p.add_argument("--root",         default="/ghome/group01/mcv/datasets/C5/KITTI-MOTS/")
    p.add_argument("--model_id",     default="facebook/sam-vit-base")
    p.add_argument("--split",        default="dev", choices=["dev", "validation"])
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--split_ratio",  type=float, default=0.8)
    p.add_argument("--prompt_type",  default="bbox", choices=["bbox", "point"])
    p.add_argument("--output",       default="results_eval/sam_pretrained_metrics.json")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
