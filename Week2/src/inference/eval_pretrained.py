"""
eval_pretrained.py

Evaluates a SAM model on the dev split with the same COCO segmentation metrics
used during finetuning. Supports two inference protocols:

  --protocol pretrained  (default)
      Per-object box format → (1, N, 1, 4), multimask_output=True, best-by-IoU,
      post_process_masks for upscaling. This is how Meta designed SAM to be used.
      Works for both pretrained and finetuned weights → always non-zero AP.

  --protocol finetuned
      Batched box format → (B, 1, N, 4), multimask_output=False, F.interpolate,
      mean-sigmoid confidence. This is the exact pipeline used during finetuning.
      Finetuned model matches best_metrics.json; pretrained model → AP≈0.

Same across both protocols: dataset, GT, DiceBCE loss, COCO AP formula.
Only the inference pipeline (box format in, mask upscaling out) differs.

Usage examples (from Week2/):
    # Pretrained weights, pretrained protocol → non-zero AP
    python -m src.eval_pretrained --protocol pretrained

    # Pretrained weights, finetuning protocol → AP≈0 (shows the protocol gap)
    python -m src.eval_pretrained --protocol finetuned

    # Finetuned weights, finetuning protocol → matches best_metrics.json
    python -m src.eval_pretrained --protocol finetuned \\
        --weights results_finetune/sam_base_lh35g5yk/best_model.pth

    # Finetuned weights, pretrained protocol → slight penalty for wrong protocol
    python -m src.eval_pretrained --protocol pretrained \\
        --weights results_finetune/sam_base_lh35g5yk/best_model.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SamModel, SamProcessor
import pycocotools.mask as rletools

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import KITTIMOTS


# ---------------------------------------------------------------------------
# Loss (same as DiceBCELoss in sam_finetune.py)
# ---------------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs_flat  = inputs.view(-1)
        targets_flat = targets.view(-1)
        bce  = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat, reduction="mean")
        sig  = torch.sigmoid(inputs_flat)
        intersection = (sig * targets_flat).sum()
        dice = 1 - (2.0 * intersection + smooth) / (sig.sum() + targets_flat.sum() + smooth)
        return bce + dice, bce, dice


# ---------------------------------------------------------------------------
# Collate (same as finetuning)
# ---------------------------------------------------------------------------
def collate_fn(batch):
    images, targets, metas = zip(*batch)
    return list(images), list(targets), list(metas)


# (prepare_batch removed — inference is now done per-image with per-object box prompts)


# ---------------------------------------------------------------------------
# Protocol: "finetuned" — mirrors prepare_batch_for_sam + flatten_preds_and_gt
# ---------------------------------------------------------------------------
def prepare_batch_finetuned(batch, processor, device):
    """Batched box format: (B, 1, N, 4), multimask_output=False — exact finetuning I/O."""
    images, batched_boxes, raw_masks_list, num_boxes = [], [], [], []
    valid_targets, valid_metas = [], []
    images_list, targets_list, metas_list = batch

    for img_pil, anns, meta in zip(images_list, targets_list, metas_list):
        boxes = [ann.bbox_xyxy for ann in anns]
        masks = [rletools.decode(ann.mask_rle).astype(np.uint8) for ann in anns]
        if not boxes:
            continue
        images.append(np.array(img_pil))
        batched_boxes.append([boxes])                # [1, N, 4]
        raw_masks_list.append(torch.tensor(np.stack(masks), dtype=torch.float32))
        num_boxes.append(len(boxes))
        valid_targets.append(anns)
        valid_metas.append(meta)

    if not images:
        return None, None, None, None, None, None

    max_n = max(num_boxes)
    for b in batched_boxes:
        while len(b[0]) < max_n:
            b[0].append([0, 0, 0, 0])

    inputs = processor(images=images, input_boxes=batched_boxes, return_tensors="pt")
    return (
        inputs["pixel_values"].to(device),
        inputs["input_boxes"].to(device),
        raw_masks_list, num_boxes, valid_metas, valid_targets,
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Protocol: {args.protocol}")

    ds = KITTIMOTS(
        root=args.root, split="dev", ann_source="txt",
        seed=args.seed, split_ratio=args.split_ratio, compute_boxes=True,
    )
    print(f"Dev split: {len(ds)} frames")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    print(f"Loading {args.model_id} …")
    processor = SamProcessor.from_pretrained(args.model_id)
    model     = SamModel.from_pretrained(args.model_id).to(device)

    if args.weights:
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {args.weights}  (missing={len(missing)}, unexpected={len(unexpected)})")

    model.eval()
    loss_fn = DiceBCELoss()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    LABEL_MAP  = ds.LABELS_MAPPING
    CATEGORIES = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]

    print("Building COCO GT …")
    gt_images, gt_annotations, ann_id = [], [], 1
    for i in range(len(ds)):
        img, anns, meta = ds[i]
        w, h = img.size
        image_id = meta["index"]
        gt_images.append({"id": image_id, "width": w, "height": h, "file_name": meta.get("image_path", "")})
        for ann in anns:
            if ann.class_id not in LABEL_MAP:
                continue
            rle  = ann.mask_rle
            area = float(rletools.area(rle))
            x1, y1, x2, y2 = ann.bbox_xyxy
            gt_annotations.append({
                "id": ann_id, "image_id": image_id,
                "category_id": LABEL_MAP[ann.class_id],
                "segmentation": rle, "bbox": [x1, y1, x2-x1, y2-y1],
                "area": area, "iscrowd": 0,
            })
            ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {"images": gt_images, "annotations": gt_annotations, "categories": CATEGORIES}
    coco_gt.createIndex()
    print(f"GT built: {len(gt_annotations)} annotations over {len(gt_images)} images")

    total_loss = total_bce = total_dice = 0.0
    n_batches  = 0
    coco_dt_list = []

    with torch.no_grad():
        # ------------------------------------------------------------------ #
        # PROTOCOL: "finetuned"                                               #
        # Batched boxes (B,1,N,4) · multimask=False · F.interpolate          #
        # Exact mirror of sam_finetune.py evaluate()                          #
        # ------------------------------------------------------------------ #
        if args.protocol == "finetuned":
            for batch in tqdm(loader, desc="Evaluating [finetuned protocol]"):
                pixel_values, input_boxes, raw_masks, num_boxes, valid_metas, valid_targets = \
                    prepare_batch_finetuned(batch, processor, device)
                if pixel_values is None:
                    continue

                outputs = model(pixel_values=pixel_values, input_boxes=input_boxes,
                                multimask_output=False)

                # pred_masks: (B, 1, N, 1, 256, 256) → view → (B, N, 256, 256)
                B = len(num_boxes)
                pred_masks = outputs.pred_masks.view(
                    B, -1, outputs.pred_masks.shape[-2], outputs.pred_masks.shape[-1]
                )

                pred_list, gt_list = [], []
                for i, n in enumerate(num_boxes):
                    pred_list.append(pred_masks[i, :n])
                    gt_i = raw_masks[i].float().to(device)
                    gt_i_r = F.interpolate(gt_i.unsqueeze(1), size=(256, 256), mode="nearest").squeeze(1)
                    gt_list.append(gt_i_r)

                pred_cat = torch.cat(pred_list).unsqueeze(1)
                gt_cat   = torch.cat(gt_list).unsqueeze(1)
                loss, bce, dice = loss_fn(pred_cat, gt_cat)
                total_loss += loss.item(); total_bce += bce.item(); total_dice += dice.item()
                n_batches  += 1

                pred_masks_cpu = pred_masks.cpu()  # (B, N, 256, 256)
                for i, (n, tgt, meta) in enumerate(zip(num_boxes, valid_targets, valid_metas)):
                    if n == 0:
                        continue
                    image_id = meta["index"]
                    raw_h, raw_w = raw_masks[i].shape[-2:]
                    logits_i = pred_masks_cpu[i, :n]  # (n, 256, 256)
                    upscaled = F.interpolate(
                        logits_i.unsqueeze(1), size=(raw_h, raw_w),
                        mode="bilinear", align_corners=False,
                    ).squeeze(1)
                    pred_binary = (torch.sigmoid(upscaled) > 0.5).numpy().astype(np.uint8)
                    scores = torch.sigmoid(upscaled).mean(dim=(-1, -2)).numpy()
                    for j in range(min(n, len(tgt))):
                        cat_id = tgt[j].class_id
                        if cat_id not in LABEL_MAP:
                            continue
                        mask_j = np.asfortranarray(pred_binary[j])
                        rle = rletools.encode(mask_j)
                        rle["counts"] = rle["counts"].decode("utf-8")
                        bbox = rletools.toBbox(rle).tolist()
                        if bbox[2] == 0 or bbox[3] == 0:
                            continue
                        coco_dt_list.append({
                            "image_id": image_id,
                            "category_id": LABEL_MAP[cat_id],
                            "segmentation": rle, "bbox": bbox,
                            "score": float(scores[j]),
                        })

        # ------------------------------------------------------------------ #
        # PROTOCOL: "pretrained"                                              #
        # Per-object boxes (1,N,1,4) · multimask=True · post_process_masks   #
        # SAM's native usage — works for both pretrained and finetuned weights#
        # ------------------------------------------------------------------ #
        else:
            for raw_batch in tqdm(loader, desc="Evaluating [pretrained protocol]"):
                images_list, targets_list, metas_list = raw_batch

                for img_pil, anns, meta in zip(images_list, targets_list, metas_list):
                    if not anns:
                        continue

                    img_np = np.array(img_pil)
                    N = len(anns)
                    boxes = [ann.bbox_xyxy for ann in anns]
                    input_boxes_fmt = [[[box] for box in boxes]]  # (1, N, 1, 4)

                    inputs = processor(images=[img_np], input_boxes=input_boxes_fmt,
                                       return_tensors="pt")
                    pv = inputs["pixel_values"].to(device)
                    ib = inputs["input_boxes"].to(device)

                    outputs = model(pixel_values=pv, input_boxes=ib, multimask_output=True)

                    # iou_scores: (1, N, 3) → pick best per object
                    iou_cpu = outputs.iou_scores.cpu()
                    best_idx   = iou_cpu[0].argmax(dim=-1)       # (N,)
                    best_scores = iou_cpu[0, torch.arange(N), best_idx]  # (N,)

                    # Loss from raw 256×256 logits (best candidate)
                    raw_logits  = outputs.pred_masks.cpu()[0]     # (N, 3, 256, 256)
                    best_logits = raw_logits[torch.arange(N), best_idx]  # (N, 256, 256)

                    gt_masks = torch.tensor(
                        np.stack([rletools.decode(ann.mask_rle).astype(np.float32) for ann in anns])
                    )
                    gt_r = F.interpolate(gt_masks.unsqueeze(1), size=(256, 256),
                                         mode="nearest").squeeze(1)
                    loss, bce, dice = loss_fn(
                        best_logits.unsqueeze(1).to(device),
                        gt_r.unsqueeze(1).to(device),
                    )
                    total_loss += loss.item(); total_bce += bce.item(); total_dice += dice.item()
                    n_batches  += 1

                    # post_process_masks → correct full-resolution masks
                    masks_pp  = processor.image_processor.post_process_masks(
                        outputs.pred_masks.cpu(),
                        inputs["original_sizes"].cpu(),
                        inputs["reshaped_input_sizes"].cpu(),
                    )
                    masks_out = masks_pp[0]  # (N, 3, H, W) bool

                    image_id = meta["index"]
                    for j, ann in enumerate(anns):
                        if ann.class_id not in LABEL_MAP:
                            continue
                        mask_j = masks_out[j, int(best_idx[j])].numpy().astype(np.uint8)
                        mask_j = np.asfortranarray(mask_j)
                        rle    = rletools.encode(mask_j)
                        rle["counts"] = rle["counts"].decode("utf-8")
                        bbox   = rletools.toBbox(rle).tolist()
                        if bbox[2] == 0 or bbox[3] == 0:
                            continue
                        coco_dt_list.append({
                            "image_id": image_id,
                            "category_id": LABEL_MAP[ann.class_id],
                            "segmentation": rle, "bbox": bbox,
                            "score": float(best_scores[j]),
                        })

    avg_loss = total_loss / max(1, n_batches)
    avg_bce  = total_bce  / max(1, n_batches)
    avg_dice = total_dice / max(1, n_batches)
    print(f"\nLoss: {avg_loss:.4f}  BCE: {avg_bce:.4f}  Dice: {avg_dice:.4f}")

    metrics = {"loss": avg_loss, "loss_bce": avg_bce, "loss_dice": avg_dice}

    if coco_dt_list:
        coco_dt = coco_gt.loadRes(coco_dt_list)

        ev = COCOeval(coco_gt, coco_dt, "segm")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        s = ev.stats
        metrics.update({
            "overall/AP_segm": s[0], "overall/AP_50_segm": s[1],
            "overall/AP_75_segm": s[2], "overall/AP_small_segm": s[3],
            "overall/AP_medium_segm": s[4], "overall/AP_large_segm": s[5],
        })
        metrics["dice"] = s[0]

        for cat in CATEGORIES:
            ev_c = COCOeval(coco_gt, coco_dt, "segm")
            ev_c.params.catIds = [cat["id"]]
            ev_c.evaluate(); ev_c.accumulate(); ev_c.summarize()
            sc = ev_c.stats
            metrics.update({
                f"{cat['name']}/AP_segm":    sc[0],
                f"{cat['name']}/AP_50_segm": sc[1],
                f"{cat['name']}/AP_75_segm": sc[2],
            })

    print(f"\n=== SAM metrics | protocol={args.protocol} | weights={'finetuned' if args.weights else 'pretrained'} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {out_path}")


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SAM (pretrained or finetuned) on dev split")
    p.add_argument("--root",         default="~/mcv/datasets/C5/KITTI-MOTS/")
    p.add_argument("--model_id",     default="facebook/sam-vit-base")
    p.add_argument("--weights",      default=None,
                   help="Path to finetuned checkpoint (.pth). Omit for pretrained weights.")
    p.add_argument("--protocol",     default="pretrained", choices=["pretrained", "finetuned"],
                   help="pretrained: per-object boxes + multimask=True + post_process_masks. "
                        "finetuned: batched boxes + multimask=False + F.interpolate.")
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--split_ratio",  type=float, default=0.8)
    p.add_argument("--output",       default="results_eval/metrics.json")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
