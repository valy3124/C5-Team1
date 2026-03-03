"""
fine_tune_detr.py — Fine-tuning script for DETR on KITTI-MOTS.

This script is intentionally scoped to **HuggingFace DETR only**.
Torchvision Faster R-CNN and Ultralytics YOLO use different training APIs and should
be fine-tuned with their own dedicated scripts:

    fine_tune.py       – Torchvision Faster R-CNN
    fine_tune_yolo.py  – Ultralytics YOLO

All three scripts share the same data pipeline (KITTI-MOTS dataset wrappers,
Albumentations augmentation, COCO evaluation) so that results are comparable.

Pipeline overview (this script):
    1. setup_experiment  – load YAML config, init W&B, create output directory.
    2. setup_data        – build KITTI-MOTS datasets + DataLoaders.
    3. setup_model       – instantiate DETR, ImageProcessor, optimizer, and scheduler.
    4. train             – run the training loop with periodic COCO evaluation.

fine_tune_detr.py — Fine-tuning script for DETR on KITTI-MOTS.
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
from transformers import DetrForObjectDetection, DetrImageProcessor
from peft import LoraConfig, get_peft_model

# Import our shared utilities
from utils import (
    Exp, Data, Run, Eval, 
    setup_experiment, setup_data, build_scheduler, 
    get_common_parser, _xyxy_to_xywh
)


def setup_model(exp: Exp, data: Data) -> Run:
    """Instantiate the HuggingFace DETR model, processor, optimizer, and scheduler."""
    cfg         = exp.cfg
    device      = exp.device
    model_name  = cfg["model"]["name"]
    weights     = cfg["model"].get("weights", "facebook/detr-resnet-50")

    # Detect if weights is a local .pth state dict vs a HuggingFace model hub name
    base_model = cfg["model"].get("base_model", "facebook/detr-resnet-50")
    is_local_pth = isinstance(weights, str) and (weights.endswith(".pth") or weights.endswith(".pt"))
    hf_source = base_model if is_local_pth else weights

    print(f"Initialising HuggingFace model: {model_name} from {hf_source}")
    if is_local_pth:
        print(f"Will load custom state dict from: {weights}")

    img_size = cfg["training"].get("image_size", 1333)
    processor = DetrImageProcessor.from_pretrained(
        hf_source,
        size={"shortest_edge": img_size, "longest_edge": img_size},
    )
    
    id2label = {i: label for i, label in enumerate(data.classes)}
    label2id = {label: i for i, label in enumerate(data.classes)}

    model = DetrForObjectDetection.from_pretrained(
        hf_source,
        num_labels=len(data.classes),
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id,
    )

    # Load custom .pth state dict on top of the HF architecture
    if is_local_pth:
        state_dict = torch.load(weights, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded state dict: {len(missing)} missing keys, {len(unexpected)} unexpected keys")

    # Either LoRA or standard freeze strategy
    lora_cfg = cfg["model"].get("lora", {})
    if lora_cfg.get("enabled", False):
        # Backbone always frozen; LoRA adapters train the encoder+decoder attention
        for param in model.model.backbone.parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.1),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias="none",
            modules_to_save=["class_labels_classifier", "bbox_predictor"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # Standard freeze strategy
        freeze_strat = cfg["training"].get("freeze_strategy", 1)
        if freeze_strat >= 1:
            for param in model.model.backbone.parameters(): param.requires_grad = False
        if freeze_strat >= 2:
            for param in model.model.encoder.parameters(): param.requires_grad = False
        if freeze_strat >= 3:
            for param in model.model.decoder.parameters(): param.requires_grad = False

    model.to(device)

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
    
    history = {"train_loss": [], "train_map": [], "val_loss": [], "val_map": []}
    return Run(model=model, processor=processor, optimizer=optimizer, history=history, scheduler=scheduler)


def evaluate(exp: Exp, run: Run, loader: Any, metrics_obj: Any, label_mapping: Dict[int, int]) -> Eval:
    """Run inference on loader and compute COCO detection metrics."""
    model, processor, device = run.model, run.processor, exp.device
    model.eval()
    coco_dt_list = []
    
    # Initialize tracking variables
    total_inference_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, targets in loader:
            batch_dict = processor(images=images, return_tensors="pt", do_rescale=False)
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            
            if device.type == "cuda": torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            outputs = model(**batch_dict)
            
            target_sizes = torch.stack([tgt["orig_size"] for tgt in targets]).to(device)
            
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)

            # Stop timer and accumulate time
            if device.type == "cuda": torch.cuda.synchronize()
            total_inference_time += (time.perf_counter() - start_time)
            total_images += len(images)

            for res, tgt in zip(results, targets):
                image_id = int(tgt["image_id"].item())

                for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
                    cat_id = int(label.item())
                    if cat_id not in label_mapping: continue

                    coco_dt_list.append({
                        "image_id":    image_id,
                        "category_id": label_mapping[cat_id],
                        "bbox":        _xyxy_to_xywh(box.cpu().tolist()),
                        "score":       float(score.item()),
                    })

    if not coco_dt_list:
        print("Warning: no predictions were generated for this split.")
        # Return 0 for speed metrics if it fails 
        return Eval(predictions=[], metrics={"coco/AP": 0.0}, inference_fps=0.0, inference_latency_ms=0.0)

    #  Calculate Latency and FPS 
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
    #Upload a sample of validation images with predicted and GT boxes to W&B.
    model, processor, device = run.model, run.processor, exp.device
    model.eval()
    
    class_names = {i: name for i, name in enumerate(data.classes)}
    
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

            subset_images = [images[i] for i in log_in_batch]
            batch_dict = processor(images=subset_images, return_tensors="pt", do_rescale=False)
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            
            outputs = model(**batch_dict)
            
            # Use orig_size to match the true dimensions of the ground truth
            target_sizes = torch.stack([targets[i].get("orig_size", torch.tensor(images[i].shape[1:])) for i in log_in_batch]).to(device)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.60) #0.25

            for res, i in zip(results, log_in_batch):
                img_np = (images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                tgt    = targets[i]

                wandb_pred_boxes = [{
                    "position": {"minX": float(b[0]), "minY": float(b[1]), "maxX": float(b[2]), "maxY": float(b[3])},
                    "class_id": int(l), "box_caption": f"{class_names.get(int(l), 'Unknown')} {s:.2f}",
                    "scores": {"score": float(s)}, "domain": "pixel"
                } for b, l, s in zip(res["boxes"], res["labels"], res["scores"])]

                wandb_gt_boxes = [{
                    "position": {"minX": float(b[0]), "minY": float(b[1]), "maxX": float(b[2]), "maxY": float(b[3])},
                    "class_id": int(l), "box_caption": f"{class_names.get(int(l), 'Unknown')} (GT)", "domain": "pixel"
                } for b, l in zip(tgt["boxes"].cpu().numpy(), tgt["labels"].cpu().numpy())]

                logged_images.append(wandb.Image(img_np, boxes={
                    "predictions":  {"box_data": wandb_pred_boxes, "class_labels": class_names},
                    "ground_truth": {"box_data": wandb_gt_boxes,   "class_labels": class_names},
                }))

            current_idx += batch_size
            if len(logged_images) >= max_images: break

    if logged_images:
        wandb.log({"val_predictions": logged_images}, commit=False)


def train(exp: Exp, data: Data, run: Run) -> Run:
    #Run the full training loop.
    num_epochs = exp.cfg["training"]["epochs"]
    print("Starting training…")

    start_train_time = time.time()
    for epoch in tqdm(range(num_epochs), desc="Epochs", mininterval=60, ascii=True):
        if exp.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(exp.device)
        # Training pass
        run.model.train()
        train_loss_sum = 0.0

        for images, targets in tqdm(data.train_loader, desc=f"Train Epoch {epoch + 1}/{num_epochs}", leave=False):
            formatted_targets = []
            for t in targets:
                anno_list = [{
                    "bbox": [box[0].item(), box[1].item(), (box[2]-box[0]).item(), (box[3]-box[1]).item()],
                    "category_id": int(label.item()), "area": float(area.item()), "iscrowd": int(iscrowd.item())
                } for box, label, area, iscrowd in zip(t["boxes"], t["labels"], t["area"], t["iscrowd"])]
                formatted_targets.append({"image_id": int(t["image_id"].item()), "annotations": anno_list})

            batch_dict = run.processor(images=images, annotations=formatted_targets, return_tensors="pt", do_rescale=False)
            
            for k, v in batch_dict.items():
                if isinstance(v, list):
                    batch_dict[k] = [{dict_k: dict_v.to(exp.device) for dict_k, dict_v in label_dict.items()} for label_dict in v]
                else:
                    batch_dict[k] = v.to(exp.device)

            outputs = run.model(**batch_dict)
            loss = outputs.loss
            
            run.optimizer.zero_grad()
            loss.backward()
            
            grad_clip = exp.cfg["training"].get("gradient_clipping", 0.1)
            torch.nn.utils.clip_grad_norm_(run.model.parameters(), max_norm=grad_clip) 
            
            run.optimizer.step()
            
            train_loss_sum += float(loss.item())

        train_loss = train_loss_sum / max(1, len(data.train_loader))

        # Validation loss pass
        run.model.eval() 
        val_loss_sum = 0.0

        with torch.no_grad():
            for images, targets in tqdm(data.val_loader, desc="Validation Pass", leave=False):
                formatted_targets = []
                for t in targets:
                    anno_list = [{
                        "bbox": [box[0].item(), box[1].item(), (box[2]-box[0]).item(), (box[3]-box[1]).item()],
                        "category_id": int(label.item()), "area": float(area.item()), "iscrowd": int(iscrowd.item())
                    } for box, label, area, iscrowd in zip(t["boxes"], t["labels"], t["area"], t["iscrowd"])]
                    formatted_targets.append({"image_id": int(t["image_id"].item()), "annotations": anno_list})

                batch_dict = run.processor(images=images, annotations=formatted_targets, return_tensors="pt", do_rescale=False)
                for k, v in batch_dict.items():
                    if isinstance(v, list):
                        batch_dict[k] = [{dict_k: dict_v.to(exp.device) for dict_k, dict_v in label_dict.items()} for label_dict in v]
                    else:
                        batch_dict[k] = v.to(exp.device)

                outputs = run.model(**batch_dict)
                val_loss_sum += float(outputs.loss.item())

        val_loss = val_loss_sum / max(1, len(data.val_loader))

        # Logging and Checkpointing
        eval_result = evaluate(exp, run, data.val_loader, data.val_coco_metrics, data.label_mapping)
        val_map = eval_result.metrics["overall/AP"]

        run.history["train_loss"].append(train_loss)
        run.history["val_loss"].append(val_loss)
        run.history["val_map"].append(val_map)

        if (epoch + 1) % exp.cfg["training"].get("log_images_every", 5) == 0:
            log_predictions_to_wandb(exp, data, run, epoch=epoch + 1)

        print(f"Epoch {epoch + 1}/{num_epochs} | train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_COCO_AP: {val_map:.4f}")

        if run.scheduler is not None: 
            run.scheduler.step()

        if val_map > run.best_map:
            run.best_map, run.best_epoch = val_map, epoch
            
            if hasattr(run.model, 'save_pretrained'):
                run.model.save_pretrained(exp.output_dir)
                torch.save(run.model.state_dict(), exp.best_model_path)
            else:
                torch.save(run.model.state_dict(), exp.best_model_path)
            
            with open(os.path.join(exp.output_dir, "best_metrics.json"), "w") as fh: 
                json.dump(eval_result.metrics, fh, indent=4)
            with open(os.path.join(exp.output_dir, "best_predictions.jsonl"), "w") as fh:
                for pred in eval_result.predictions:
                    fh.write(json.dumps(pred) + "\n")
            print(f"New best: COCO AP {run.best_map:.4f} at epoch {epoch + 1}")

        peak_vram_gb = torch.cuda.max_memory_allocated(exp.device) / 1024**3 if exp.device.type == "cuda" else 0
        wandb_log = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_coco_ap": val_map, "best_map": run.best_map, "best_epoch": run.best_epoch, "inference_fps": eval_result.inference_fps,"inference_latency_ms": eval_result.inference_latency_ms, "peak_vram_gb": peak_vram_gb}
        wandb_log.update({f"val_{k}": v for k, v in eval_result.metrics.items()})
        wandb.log(wandb_log)

        csv_path = os.path.join(exp.output_dir, "metrics_history.csv")
        write_header = not os.path.isfile(csv_path)
        headers = ["epoch", "train_loss", "val_loss", "best_map"] + [f"val_{k}" for k in eval_result.metrics.keys()]
        with open(csv_path, "a") as fh:
            if write_header: fh.write(",".join(headers) + "\n")
            row = [str(epoch + 1), f"{train_loss:.6f}", f"{val_loss:.6f}", f"{run.best_map:.6f}"] + [f"{v:.6f}" for v in eval_result.metrics.values()]
            fh.write(",".join(row) + "\n")

    print(f"Training complete. Best COCO AP: {run.best_map:.4f} at epoch {run.best_epoch + 1}.")
    total_train_time_hrs = (time.time() - start_train_time) / 3600
    wandb.run.summary["total_train_time_hrs"] = total_train_time_hrs
    print(f"Total Training Time: {total_train_time_hrs:.2f} hours")
    return run


def main(args):
    exp  = setup_experiment(args.config, args)
    data = setup_data(exp)
    
    # Tag the run name
    if args.eval_only:
        exp.cfg["experiment_name"] += "_ZERO_SHOT"
        
    wandb.run.name = exp.cfg["experiment_name"]
    
    run  = setup_model(exp, data)
    
    if args.eval_only:
        print("Running Zero-Shot Evaluation")
        eval_result = evaluate(exp, run, data.val_loader, data.val_coco_metrics, data.label_mapping)
        
        print(f"\nZero-Shot COCO AP: {eval_result.metrics['overall/AP']:.4f}")
        
        # Log 8 images to WandB to visualise 
        log_predictions_to_wandb(exp, data, run, epoch=0, max_images=8)
        
        # Log metrics to WandBS
        wandb_log = {f"zero_shot_val_{k}": v for k, v in eval_result.metrics.items()}
        wandb_log["inference_fps"] = eval_result.inference_fps
        wandb.log(wandb_log)
    else:
        # Standard training loop
        run  = train(exp, data, run)
        
    wandb.finish()


if __name__ == "__main__":
    parser = get_common_parser("Fine-tune DETR on KITTI-MOTS.")
    main(parser.parse_args())