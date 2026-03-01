"""
fine_tune.py — Fine-tuning script for Faster R-CNN on KITTI-MOTS.

This script is intentionally scoped to **torchvision Faster R-CNN only**.
HuggingFace DETR and Ultralytics YOLO use different training APIs and should
be fine-tuned with their own dedicated scripts:

    fine_tune_detr.py  – HuggingFace Transformers / DETR
    fine_tune_yolo.py  – Ultralytics YOLO

All three scripts share the same data pipeline (KITTI-MOTS dataset wrappers,
Albumentations augmentation, COCO evaluation) so that results are comparable.

Pipeline overview (this script):
    1. setup_experiment  – load YAML config, init W&B, create output directory.
    2. setup_data        – build KITTI-MOTS datasets + DataLoaders.
    3. setup_model       – instantiate Faster R-CNN, optimizer, and scheduler.
    4. train             – run the training loop with periodic COCO evaluation.

fine_tune.py — Fine-tuning script for Torchvision Faster R-CNN on KITTI-MOTS.
Uses the shared pipeline from utils.py.
"""

import os
import json
import torch
import torch.optim as optim
import wandb
import time
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List

# Import our shared utilities
from utils import (
    Exp, Data, Run, Eval, 
    setup_experiment, setup_data, build_scheduler, 
    get_common_parser, _xyxy_to_xywh
)

from models.faster_rcnn import FasterRCNNModel

# Label mapping from model class IDs to COCO category IDs.
# Model:  1 = Car,    2 = Pedestrian
# COCO:   3 = Car,    1 = Person
_MODEL_TO_COCO_LABEL = {1: 3, 2: 1}
CLASS_NAMES = {1: "Car", 2: "Pedestrian"}


def setup_model(exp: Exp, data: Data) -> Run:
    """Instantiate the Faster R-CNN model, optimizer, and scheduler."""
    cfg, device = exp.cfg, exp.device
    model_name = cfg["model"]["name"]

    print(f"Initialising model: {model_name}")

    if model_name != "faster_rcnn":
        raise ValueError("This script only supports 'faster_rcnn'. Use fine_tune_detr.py for DETR.")

    nms_iou      = cfg["training"].get("nms_iou_threshold", 0.5)
    model        = FasterRCNNModel(device=str(device), iou=nms_iou)
    num_classes  = len(data.classes)
    freeze_strat = cfg["training"].get("freeze_strategy", 1)
    
    # Prepare the model head and freeze specific layers based on strategy
    model.prepare_finetune(num_classes, freeze_strat)

    # Only pass parameters that require gradients to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    opt_name = cfg["training"].get("optimizer", "adamw").lower()
    if opt_name == "adamw":
        optimizer = optim.AdamW(trainable_params, lr=cfg["training"]["lr"])
    elif opt_name == "adam":
        optimizer = optim.Adam(trainable_params, lr=cfg["training"]["lr"])
    elif opt_name == "sgd":
        optimizer = optim.SGD(trainable_params, lr=cfg["training"]["lr"], momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer '{opt_name}'")
        
    scheduler = build_scheduler(optimizer, cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"total_params": total_params, "trainable_params": trainable_params}, allow_val_change=True)
    print(f"Model Parameters: Total={total_params:,} | Trainable={trainable_params:,}")
    
    history = {"train_loss": [], "train_map": [], "val_map": []}
    
    # processor is None because Faster R-CNN doesn't need HuggingFace ImageProcessors
    return Run(model=model, optimizer=optimizer, history=history, scheduler=scheduler)


def evaluate(exp: Exp, run: Run, loader: Any, metrics_obj: Any) -> Eval:
    """Run inference on loader and compute COCO detection metrics."""
    model, device = run.model, exp.device
    model.eval()
    coco_dt_list = []

    total_inference_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, targets in loader:
            imgs  = [img.to(device) for img in images]
            
            if device.type == "cuda": torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            preds, _ = model.predict(imgs)  # Returns List[Dict[bboxes_xyxy, category_ids, scores]]

            #  Stop timer and accumulate time 
            if device.type == "cuda": torch.cuda.synchronize()
            total_inference_time += (time.perf_counter() - start_time)
            total_images += len(images)

            for pred, tgt in zip(preds, targets):
                image_id = int(tgt["image_id"].item())

                for box, cat, score in zip(pred["bboxes_xyxy"], pred["category_ids"], pred["scores"]):
                    cat_id = int(cat)
                    if cat_id not in _MODEL_TO_COCO_LABEL:
                        continue 

                    coco_dt_list.append({
                        "image_id":    image_id,
                        "category_id": _MODEL_TO_COCO_LABEL[cat_id],
                        "bbox":        _xyxy_to_xywh(box),
                        "score":       float(score),
                    })

    if not coco_dt_list:
        print("Warning: no predictions were generated for this split.")
        # eturn 0 for speed metrics if it fails 
        return Eval(predictions=[], metrics={"coco/AP": 0.0}, inference_fps=0.0, inference_latency_ms=0.0)

    # - Calculate Latency and FPS 
    avg_latency_ms = (total_inference_time / total_images) * 1000 if total_images > 0 else 0.0
    fps = total_images / total_inference_time if total_inference_time > 0 else 0.0

    coco_dt = metrics_obj.coco_gt.loadRes(coco_dt_list)

    return Eval(
        predictions=coco_dt_list, 
        metrics=metrics_obj.compute_metrics(coco_dt),
        inference_fps=fps,
        inference_latency_ms=avg_latency_ms
    )


def log_predictions_to_wandb(exp: Exp, data: Data, run: Run, epoch: int, max_images: int = 4) -> None:
    """Upload a sample of validation images with predicted and GT boxes to W&B."""
    model, device = run.model, exp.device
    model.eval()

    dataset_len = len(data.val_loader.dataset)
    step = max(1, dataset_len // max_images)
    target_indices = set(range(0, dataset_len, step))
    logged_images, current_idx = [], 0

    with torch.no_grad():
        for images, targets in data.val_loader:
            batch_size = len(images)
            log_in_batch = [i for i in range(batch_size) if current_idx + i in target_indices][:max_images - len(logged_images)]

            if not log_in_batch:
                current_idx += batch_size
                continue

            subset_images = [images[i].to(device) for i in log_in_batch]
            preds, _ = model.predict(subset_images)

            for pred, i in zip(preds, log_in_batch):
                img_np = (images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                tgt    = targets[i]

                wandb_pred_boxes = [{
                    "position": {"minX": float(b[0]), "minY": float(b[1]), "maxX": float(b[2]), "maxY": float(b[3])},
                    "class_id": int(l), "box_caption": f"{CLASS_NAMES.get(int(l), 'Unknown')} {s:.2f}",
                    "scores": {"score": float(s)}, "domain": "pixel"
                } for b, l, s in zip(pred["bboxes_xyxy"], pred["category_ids"], pred["scores"]) if s >= 0.25]

                wandb_gt_boxes = [{
                    "position": {"minX": float(b[0]), "minY": float(b[1]), "maxX": float(b[2]), "maxY": float(b[3])},
                    "class_id": int(l), "box_caption": f"{CLASS_NAMES.get(int(l), 'Unknown')} (GT)", "domain": "pixel"
                } for b, l in zip(tgt["boxes"].cpu().numpy(), tgt["labels"].cpu().numpy())]

                logged_images.append(wandb.Image(img_np, boxes={
                    "predictions":  {"box_data": wandb_pred_boxes, "class_labels": CLASS_NAMES},
                    "ground_truth": {"box_data": wandb_gt_boxes,   "class_labels": CLASS_NAMES},
                }))

            current_idx += batch_size
            if len(logged_images) >= max_images: break

    if logged_images:
        wandb.log({"val_predictions": logged_images}, commit=False)


def train(exp: Exp, data: Data, run: Run) -> Run:
    """Run the full training loop."""
    num_epochs = exp.cfg["training"]["epochs"]
    print("Starting Faster R-CNN training…")

    start_train_time = time.time()
    for epoch in tqdm(range(num_epochs), desc="Epochs", mininterval=60, ascii=True):
        if exp.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(exp.device)
            
        # ---- Training pass ----
        run.model.train()
        train_loss_sum = 0.0

        for images, targets in data.train_loader:
            images  = [img.to(exp.device) for img in images]
            targets = [{k: v.to(exp.device) for k, v in t.items()} for t in targets]

            # Torchvision Faster R-CNN natively returns a dictionary of losses during training
            loss_dict = run.model(images, targets) 
            losses    = sum(loss_dict.values())

            run.optimizer.zero_grad()
            losses.backward()
            
            grad_clip = exp.cfg["training"].get("gradient_clipping")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(run.model.parameters(), grad_clip)
                
            run.optimizer.step()

            train_loss_sum += float(losses.item())

        train_loss = train_loss_sum / max(1, len(data.train_loader))

        # ---- Evaluation pass ----
        eval_result = evaluate(exp, run, data.val_loader, data.val_coco_metrics)
        val_map = eval_result.metrics["overall/AP"]

        run.history["train_loss"].append(train_loss)
        run.history["val_map"].append(val_map)

        if (epoch + 1) % exp.cfg["training"].get("log_images_every", 5) == 0:
            log_predictions_to_wandb(exp, data, run, epoch=epoch + 1)

        print(f"Epoch {epoch + 1}/{num_epochs} | train_loss: {train_loss:.4f} val_COCO_AP: {val_map:.4f}")

        if run.scheduler is not None:
            run.scheduler.step()

        # ---- Checkpoint best model ----
        if val_map > run.best_map:
            run.best_map, run.best_epoch = val_map, epoch
            torch.save(run.model.state_dict(), exp.best_model_path)
            
            with open(os.path.join(exp.output_dir, "best_metrics.json"), "w") as fh:
                json.dump(eval_result.metrics, fh, indent=4)
            with open(os.path.join(exp.output_dir, "best_predictions.jsonl"), "w") as fh:
                for pred in eval_result.predictions:
                    fh.write(json.dumps(pred) + "\n")
            print(f"  >>> New best: COCO AP {run.best_map:.4f} at epoch {epoch + 1}")

        # ---- Logging ----
        peak_vram_gb = torch.cuda.max_memory_allocated(exp.device) / 1024**3 if exp.device.type == "cuda" else 0
        wandb_log = {"epoch": epoch + 1, "train_loss": train_loss, "val_coco_ap": val_map, "best_map": run.best_map, "best_epoch": run.best_epoch, "inference_fps": eval_result.inference_fps, "inference_latency_ms": eval_result.inference_latency_ms, "peak_vram_gb": peak_vram_gb}
        wandb_log.update({f"val_{k}": v for k, v in eval_result.metrics.items()})
        wandb.log(wandb_log)

        csv_path = os.path.join(exp.output_dir, "metrics_history.csv")
        write_header = not os.path.isfile(csv_path)
        headers = ["epoch", "train_loss", "best_map"] + [f"val_{k}" for k in eval_result.metrics.keys()]
        
        with open(csv_path, "a") as fh:
            if write_header: fh.write(",".join(headers) + "\n")
            row = [str(epoch + 1), f"{train_loss:.6f}", f"{run.best_map:.6f}"] + [f"{v:.6f}" for v in eval_result.metrics.values()]
            fh.write(",".join(row) + "\n")

    print(f"Training complete. Best COCO AP: {run.best_map:.4f} at epoch {run.best_epoch + 1}.")
    total_train_time_hrs = (time.time() - start_train_time) / 3600
    wandb.run.summary["total_train_time_hrs"] = total_train_time_hrs
    print(f"Total Training Time: {total_train_time_hrs:.2f} hours")
    return run


def main(args):
    exp  = setup_experiment(args.config, args)
    data = setup_data(exp)
    run  = setup_model(exp, data)
    run  = train(exp, data, run)
    wandb.finish()


if __name__ == "__main__":
    parser = get_common_parser("Fine-tune Faster R-CNN on KITTI-MOTS.")
    main(parser.parse_args())
