#!/usr/bin/env python
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb
from pathlib import Path

# Setup paths to import project modules
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from Week2.src.datasets import KITTIMOTS, InstanceAnn
import pycocotools.mask as rletools
from transformers import SamModel, SamProcessor, logging as hf_logging

hf_logging.set_verbosity_error()

import albumentations as A
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='albumentations.augmentations.dropout.functional')

from Week2.src.finetune.utils import (
    Exp, Data, Run, Eval, 
    setup_experiment, build_scheduler, get_common_parser
)
from Week2.src.inference.evaluation_segm import CocoSegmentationMetrics
from typing import Any
import PIL.Image as Image
# -----------------------------------------------------------------------------
# SAM Custom Dataset & Collate
# -----------------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate BCE Loss with Logits
        BCE = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat, reduction='mean')
        
        # Apply sigmoid to raw logits strictly for Dice calculation
        inputs_sig = F.sigmoid(inputs_flat)       
        
        # Calculate Dice Loss
        intersection = (inputs_sig * targets_flat).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_sig.sum() + targets_flat.sum() + smooth)  
        
        # Combine them
        return BCE + dice_loss, BCE, dice_loss

def collate_fn(batch):
    images, targets, metas = zip(*batch)
    # Return them all as lists
    return list(images), list(targets), list(metas)

# -----------------------------------------------------------------------------
# Data Augmentations
# -----------------------------------------------------------------------------
def get_segm_transforms(is_train: bool = True, aug_strategy: str = "base") -> A.Compose:
    bbox_params = A.BboxParams(format="pascal_voc", label_fields=["class_labels"], clip=True, min_area=1, min_visibility=0.1)
    tfms = [] 

    if not is_train or aug_strategy == "no_aug":
        return A.Compose(tfms, bbox_params=bbox_params)

    if aug_strategy == "legacy":
        tfms += [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, border_mode=0, p=0.5),
            A.OneOf([
                A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1),
                A.RandomFog(fog_coef_range=(0.3, 1), alpha_coef=0.08, p=1),
                A.RandomShadow(p=1),
            ], p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(p=0.3),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, p=0.3),
            A.MotionBlur(p=0.2),
        ]
    else:
        tfms += [A.HorizontalFlip(p=0.5)]
        if aug_strategy in ("geometric", "limit_test"):
            tfms += [A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15, border_mode=0, p=0.5), A.Perspective(scale=(0.05, 0.1), p=0.3)]
        if aug_strategy in ("color_jitter", "limit_test"):
            tfms += [A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5), A.HueSaturationValue(p=0.3), A.RandomGamma(gamma_limit=(80, 120), p=0.3), A.RGBShift(p=0.2)]
        if aug_strategy in ("extreme_weather", "limit_test"):
            prob = 0.5 if aug_strategy == "limit_test" else 0.8
            tfms += [A.OneOf([A.RandomRain(brightness_coefficient=0.7, drop_width=1, blur_value=5, p=1), A.RandomFog(fog_coef_range=(0.4, 0.9), alpha_coef=0.1, p=1), A.RandomShadow(shadow_limit=(1, 3), p=1), A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_limit=(0, 1), p=1)], p=prob)]
        if aug_strategy in ("heavy_corruption", "limit_test"):
            tfms += [A.GaussNoise(p=0.3), A.MotionBlur(blur_limit=(3, 7), p=0.3), A.CoarseDropout(num_holes_range=(2, 10), hole_height_range=(10, 40), hole_width_range=(10, 40), fill=0, p=0.4), A.ImageCompression(quality_range=(50, 90), p=0.3)]
    
    return A.Compose(tfms, bbox_params=bbox_params)

class ApplyAlbumentationsSegm(Dataset):
    """Wraps KITTIMOTS data with Albumentations, passing mask+bbox annotations."""
    def __init__(self, ds: Dataset, tf: A.Compose) -> None:
        self.ds = ds
        self.tf = tf

    def __len__(self) -> int:
        return len(self.ds)
        
    def _bbox_from_binary(self, mask: np.ndarray) -> tuple:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def __getitem__(self, idx: int):
        img_pil, raw_anns, meta = self.ds[idx]
        img_np = np.array(img_pil)
        
        if len(raw_anns) == 0:
            # If no annots to begin with, just pass through identity to prevent crashes
            return img_pil, raw_anns, meta
            
        boxes = []
        labels = []
        masks = []
        
        # Decode the RLE masks into dense format for the image-level transform
        for ann in raw_anns:
            x1, y1, x2, y2 = ann.bbox_xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))
            labels.append(ann.class_id)
            masks.append(rletools.decode(ann.mask_rle).astype(np.uint8))
            
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))
            
        out = self.tf(image=img_np, bboxes=boxes, class_labels=labels, masks=masks)
        
        aug_img_np = out["image"]
        aug_boxes = out["bboxes"]
        aug_labels = out["class_labels"]
        aug_masks = out["masks"]
        
        if len(aug_boxes) == 0:
             # Fast-fail recovery: target disappeared. Fetch a safe unaugmented adjacent item randomly
             return self.__getitem__((idx + 1) % len(self))
             
        # Re-pack the transformed items into the original InstanceAnn schema
        new_anns = []
        for orig_ann, aug_mask, aug_box, class_label in zip(raw_anns, aug_masks, aug_boxes, aug_labels):
            aug_mask = np.asfortranarray(aug_mask)
            rle = rletools.encode(aug_mask)
            rle['counts'] = rle['counts'].decode('utf-8')
            
            # Use Albumentations transformed bounding box, clamped just in case
            h, w = aug_img_np.shape[:2]
            x1, y1, x2, y2 = aug_box
            safe_box = (max(0, min(w, x1)), max(0, min(h, y1)), max(0, min(w, x2)), max(0, min(h, y2)))
            
            new_anns.append(
                InstanceAnn(
                    object_id=orig_ann.object_id,
                    class_id=class_label,
                    instance_id=orig_ann.instance_id,
                    mask_rle=rle,
                    bbox_xyxy=safe_box
                )
            )

        aug_pil = Image.fromarray(aug_img_np)
        return aug_pil, new_anns, meta

# -----------------------------------------------------------------------------
# Pipeline Components (setup_data, setup_model, train, evaluate)
# -----------------------------------------------------------------------------
def setup_data(exp: Exp) -> Data:
    cfg = exp.cfg
    root = cfg["data"]["root"]
    mode = cfg["data"].get("mode", "full")
    seed = cfg["data"].get("seed", 42)
    split_ratio = cfg["data"].get("split_ratio", 0.8)
    aug_strategy = cfg["training"].get("aug_strategy", "legacy")

    if mode == "full": train_split, val_split = "train_full", "validation"
    elif mode == "search": train_split, val_split = "train", "dev"
    else: raise ValueError(f"Unknown data.mode '{mode}'.")

    train_ds = KITTIMOTS(root, split=train_split, ann_source="txt", seed=seed, split_ratio=split_ratio)
    val_ds   = KITTIMOTS(root, split=val_split, ann_source="txt", seed=seed, split_ratio=split_ratio)
    
    # Wrap train_ds with tracking augmentations
    if aug_strategy != "no_aug":
        train_ds = ApplyAlbumentationsSegm(train_ds, get_segm_transforms(True, aug_strategy))
        
    classes    = ["Background", "Car", "Pedestrian"]

    num_workers = cfg["data"].get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"].get("val_batch_size", 4), shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    dataset_wrapper = cfg["data"].get("dataset_wrapper", "kitti_mots").lower()
    
    val_coco_metrics = CocoSegmentationMetrics(
        root=root,
        dataset_name=dataset_wrapper,
        split=val_split,
        ann_source="txt" if dataset_wrapper == "kitti_mots" else "xml",
        seed=seed,
        split_ratio=split_ratio
    )

    print(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")

    return Data(classes, train_loader, val_loader, None, val_coco_metrics, None)

def setup_model(exp: Exp, data: Data) -> Run:
    cfg = exp.cfg
    device = exp.device
    
    model_id = cfg["model"].get("model_id", "facebook/sam-vit-base")
    print(f"Initialising SAM Model: {model_id}")
    
    model = SamModel.from_pretrained(model_id).to(device)
    processor = SamProcessor.from_pretrained(model_id)
    
    # We only fine-tune the Mask Decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
            
    # Optimizer and Scheduler
    lr = cfg["training"].get("lr", 1e-5)
    weight_decay = cfg["training"].get("weight_decay", 1e-4)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, cfg)
    
    total_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in trainable_params)
    wandb.config.update({"total_params": total_params, "trainable_params": num_trainable}, allow_val_change=True)
    print(f"Model Parameters: Total={total_params:,} | Trainable={num_trainable:,}")

    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    return Run(
        model=model, 
        optimizer=optimizer, 
        history=history, 
        scheduler=scheduler,
        processor=processor
    )

def prepare_batch_for_sam(batch, processor, device):
    """Process a raw batch of KITTIMOTS tuples into SAM inputs."""
    images = []
    batched_input_boxes = []
    raw_masks_list = []
    num_boxes = []
    
    images_list, targets_list, metas_list = batch
    
    valid_targets_list = []
    valid_metas_list = []
    
    for img_pil, anns, meta in zip(images_list, targets_list, metas_list):
        img_np = np.array(img_pil)
        
        boxes = []
        masks = []
        for ann in anns:
            boxes.append(ann.bbox_xyxy)
            masks.append(rletools.decode(ann.mask_rle).astype(np.uint8))
            
        if len(boxes) == 0:
            continue
            
        images.append(img_np)
        batched_input_boxes.append([boxes])
        raw_masks_list.append(torch.tensor(np.stack(masks), dtype=torch.float32))
        num_boxes.append(len(boxes))
        valid_targets_list.append(anns)
        valid_metas_list.append(meta)

    if len(images) == 0:
        return None, None, None, None, None, None, None, None
        
    # SamProcessor expects homogeneously sized lists or crashes when converting to np.array internally
    max_boxes = max(num_boxes)
    for boxes in batched_input_boxes:
        while len(boxes[0]) < max_boxes:
            boxes[0].append([0, 0, 0, 0])
            
    inputs = processor(
        images=images,
        input_boxes=batched_input_boxes,
        return_tensors="pt"
    )
    
    pixel_values = inputs["pixel_values"].to(device)
    input_boxes = inputs["input_boxes"].to(device)
    original_sizes = inputs["original_sizes"].to(device)
    reshaped_input_sizes = inputs["reshaped_input_sizes"].to(device)
    
    return pixel_values, input_boxes, raw_masks_list, num_boxes, valid_metas_list, valid_targets_list, original_sizes, reshaped_input_sizes


def postprocess_preds_and_flatten(outputs, raw_masks, num_boxes, original_sizes, reshaped_input_sizes, device):
    """Upsample pred_masks to original image sizes and flatten for loss computation."""
    B = len(num_boxes)
    pred_masks = outputs.pred_masks.view(B, -1, outputs.pred_masks.shape[-2], outputs.pred_masks.shape[-1])
    pred_ious_out = outputs.iou_scores.view(B, -1)

    pred_list, gt_list, real_ious, pred_ious = [], [], [], []
    for i, n in enumerate(num_boxes):
        if n == 0:
            pred_list.append(None)
            continue
            
        valid_preds = pred_masks[i, :n].unsqueeze(1) # (n, 1, 256, 256)
        orig_h, orig_w = original_sizes[i].tolist()
        reshaped_h, reshaped_w = reshaped_input_sizes[i].tolist()
        
        # 1) Upsample to 1024x1024
        up_masks = F.interpolate(valid_preds, size=(1024, 1024), mode="bilinear", align_corners=False)
        # 2) Crop pad
        up_masks = up_masks[..., :reshaped_h, :reshaped_w]
        # 3) Upsample to original size
        up_masks = F.interpolate(up_masks, size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(1)
        
        gt_mask = raw_masks[i].float().to(device)
        pred_list.append(up_masks) # (n, orig_h, orig_w)
        gt_list.append(gt_mask) # (n, orig_h, orig_w)
        
        # Compute true IoU for MSE loss
        with torch.no_grad():
            pred_bin = (up_masks > 0).float()
            inter = (pred_bin * gt_mask).sum(dim=(-1, -2))
            union = pred_bin.sum(dim=(-1, -2)) + gt_mask.sum(dim=(-1, -2)) - inter
            real_ious.append(inter / (union + 1e-6))
            
        pred_ious.append(pred_ious_out[i, :n])

    pred_1d = torch.cat([p.flatten() for p in pred_list if p is not None]) if pred_list else torch.empty(0, device=device)
    gt_1d = torch.cat([g.flatten() for g in gt_list]) if gt_list else torch.empty(0, device=device)
    real_ious_1d = torch.cat(real_ious) if real_ious else torch.empty(0, device=device)
    pred_ious_1d = torch.cat(pred_ious) if pred_ious else torch.empty(0, device=device)

    return pred_1d, gt_1d, pred_list, real_ious_1d, pred_ious_1d

def evaluate(exp: Exp, run: Run, loader: DataLoader, coco_metrics: Any = None) -> Eval:
    """Evaluates SAM model computing average Dice Loss & BCE and COCO SEG metrics"""
    model = run.model
    device = exp.device
    model.eval()
    
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    seg_loss_fn = DiceBCELoss()
    coco_dt_list = []
    
    with torch.no_grad():
        for batch in loader:
            pixel_values, input_boxes, raw_masks, num_boxes, valid_metas_list, valid_targets_list, original_sizes, reshaped_input_sizes = prepare_batch_for_sam(batch, run.processor, device)
            if pixel_values is None:
                continue
            
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(
                    pixel_values=pixel_values,
                    input_boxes=input_boxes,
                    multimask_output=False
                )
                pred_1d, gt_1d, pred_list, real_ious_1d, pred_ious_1d = postprocess_preds_and_flatten(outputs, raw_masks, num_boxes, original_sizes, reshaped_input_sizes, device)
                
                loss_seg, loss_bce, loss_dice = seg_loss_fn(pred_1d, gt_1d)
                loss_iou = F.mse_loss(pred_ious_1d, real_ious_1d)
                loss = loss_seg + loss_iou
                
            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_dice += loss_dice.item()

            if coco_metrics is not None:
                iou_scores_out = outputs.iou_scores.view(len(num_boxes), -1).cpu()
                for i, (n, tgt, meta) in enumerate(zip(num_boxes, valid_targets_list, valid_metas_list)):
                    if n == 0 or pred_list[i] is None: continue
                    image_id = meta["index"]
                    
                    mask_logits_i_resized = pred_list[i].cpu() # (n, raw_h, raw_w)
                    iou_scores_i = iou_scores_out[i, :n].numpy()
                    
                    pred_binary = (torch.sigmoid(mask_logits_i_resized) > 0.5).numpy().astype(np.uint8)
                    scores = iou_scores_i
                    
                    # Ensure we don't index past the targets list if augmentations dropped any
                    safe_n = min(n, len(tgt))
                    for j in range(safe_n):
                        cat_id = tgt[j].class_id
                        if cat_id not in coco_metrics.label_map: continue
                        
                        coco_cat_id = coco_metrics.label_map[cat_id]
                        mask_j = np.asfortranarray(pred_binary[j])
                        rle = rletools.encode(mask_j)
                        rle['counts'] = rle['counts'].decode('utf-8')
                        bbox = rletools.toBbox(rle).tolist()
                        
                        coco_dt_list.append({
                            "image_id": image_id,
                            "category_id": coco_cat_id,
                            "segmentation": rle,
                            "bbox": bbox,
                            "score": float(scores[j])
                        })

    avg_loss = total_loss / max(1, len(loader))
    avg_bce = total_bce / max(1, len(loader))
    avg_dice = total_dice / max(1, len(loader))
    
    metrics = {
        "loss": avg_loss,
        "loss_bce": avg_bce,
        "loss_dice": avg_dice
    }
    
    if coco_metrics is not None and len(coco_dt_list) > 0:
        coco_dt = coco_metrics.coco_gt.loadRes(coco_dt_list)
        coco_res = coco_metrics.compute_metrics(coco_dt)
        metrics.update(coco_res)
        metrics["dice"] = metrics.get("overall/AP_segm", 0.0)
    else:
        metrics["dice"] = avg_loss
    
    return Eval(predictions=coco_dt_list, metrics=metrics, inference_fps=0, inference_latency_ms=0)

def train(exp: Exp, data: Data, run: Run) -> Run:
    num_epochs = exp.cfg["training"]["epochs"]
    device = exp.device
    
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    seg_loss_fn = DiceBCELoss()
    
    print("Starting SAM Mask-Decoder training…")
    
    pbar = tqdm(range(num_epochs), desc="Epochs", ascii=True)
    for epoch in pbar:
        # Training pass
        run.model.train()
        train_loss_sum = 0.0
        train_bce_sum = 0.0
        train_dice_sum = 0.0
        
        for batch_idx, batch in enumerate(data.train_loader):
            pixel_values, input_boxes, raw_masks, num_boxes, _, _, original_sizes, reshaped_input_sizes = prepare_batch_for_sam(batch, run.processor, device)
            
            if pixel_values is None:
                continue
            
            run.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                outputs = run.model(
                    pixel_values=pixel_values,
                    input_boxes=input_boxes,
                    multimask_output=False
                )
                pred_1d, gt_1d, _, real_ious_1d, pred_ious_1d = postprocess_preds_and_flatten(outputs, raw_masks, num_boxes, original_sizes, reshaped_input_sizes, device)
                
                loss_seg, loss_bce, loss_dice = seg_loss_fn(pred_1d, gt_1d)
                loss_iou = F.mse_loss(pred_ious_1d, real_ious_1d)
                loss = loss_seg + loss_iou
                
            scaler.scale(loss).backward()
            
            grad_clip = exp.cfg["training"].get("gradient_clipping")
            if grad_clip is not None:
                scaler.unscale_(run.optimizer)
                torch.nn.utils.clip_grad_norm_(run.model.parameters(), grad_clip)
                
            scaler.step(run.optimizer)
            scaler.update()
            
            train_loss_sum += loss.item()
            train_bce_sum += loss_bce.item()
            train_dice_sum += loss_dice.item()

        train_loss = train_loss_sum / max(1, len(data.train_loader))
        train_bce = train_bce_sum / max(1, len(data.train_loader))
        train_dice = train_dice_sum / max(1, len(data.train_loader))
        
        # Evaluation pass
        print(f"Evaluating epoch {epoch + 1}/{num_epochs}")
        eval_result = evaluate(exp, run, data.val_loader, data.val_coco_metrics)
        
        # Validation Metrics
        val_segm_ap = eval_result.metrics.get("overall/AP_segm", 0.0) 
        val_loss = eval_result.metrics.get("loss", 0.0)
        val_bce = eval_result.metrics.get("loss_bce", 0.0)
        val_dice = eval_result.metrics.get("loss_dice", 0.0)
        
        run.history["train_loss"].append(train_loss)
        
        if val_segm_ap > run.best_map:
            run.best_map, run.best_epoch = val_segm_ap, epoch
            torch.save(run.model.state_dict(), exp.best_model_path)
            
            with open(os.path.join(exp.output_dir, "best_metrics.json"), "w") as fh:
                json.dump(eval_result.metrics, fh, indent=4)
            print(f"  >>> New best: COCO AP_segm {run.best_map:.4f} at epoch {epoch + 1}")
            
        print(f"Epoch {epoch + 1}/{num_epochs} | train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_COCO_AP_segm: {val_segm_ap:.4f}")

        wandb_log = {
            "epoch": epoch + 1, 
            "train_loss": train_loss, 
            "train_bce": train_bce,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_bce": val_bce,
            "val_dice": val_dice,
            "map_segm": val_segm_ap,
            "best_map_segm": run.best_map, 
            "best_epoch": run.best_epoch
        }
        wandb_log.update({f"val_{k}": v for k, v in eval_result.metrics.items()})
        wandb.log(wandb_log)

        csv_path = os.path.join(exp.output_dir, "metrics_history.csv")
        write_header = not os.path.isfile(csv_path)
        headers = ["epoch", "train_loss", "best_map_segm"] + [f"val_{k}" for k in eval_result.metrics.keys()]
        
        with open(csv_path, "a") as fh:
            if write_header: fh.write(",".join(headers) + "\n")
            row = [str(epoch + 1), f"{train_loss:.6f}", f"{run.best_map:.6f}"] + [f"{v:.6f}" for v in eval_result.metrics.values()]
            fh.write(",".join(row) + "\n")
            
        if run.scheduler is not None:
            run.scheduler.step()
            
    print(f"Training complete. Best COCO AP_segm: {run.best_map:.4f} at epoch {run.best_epoch + 1}.")
    return run

def main(args):
    print("Setting up experiment...")
    exp  = setup_experiment(args.config, args)
    print("Setting up data...")
    data = setup_data(exp)
    print("Setting up model...")
    run  = setup_model(exp, data)
    print("Training...")
    run  = train(exp, data, run)
    print("Training complete.")
    wandb.finish()

if __name__ == "__main__":
    parser = get_common_parser("Fine-tune SAM on KITTI-MOTS.")
    main(parser.parse_args())
