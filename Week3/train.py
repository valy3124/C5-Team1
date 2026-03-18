import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import time
import json
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import wandb
import shutil
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from dataset import VizWizDataset, idx2char, char2idx
from model import ImageCaptioningModel, ENCODER_CONFIGS

# --- Paths ---
DATA_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'VizWiz')
TRAIN_ANN = os.path.join(DATA_BASE_DIR, 'annotations', 'train.json')
VAL_ANN = os.path.join(DATA_BASE_DIR, 'annotations', 'val.json')
TRAIN_IMG_DIR = os.path.join(DATA_BASE_DIR, 'images', 'train')
VAL_IMG_DIR = os.path.join(DATA_BASE_DIR, 'images', 'val')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load HF evaluate metrics
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')
try:
    cider = evaluate.load('cider')
    CIDER_AVAILABLE = True
except Exception as e:
    cider = None
    CIDER_AVAILABLE = False
    print(f"Warning: CIDEr metric unavailable ({e}). CIDEr will not be logged.")


import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Train image captioning model on VizWiz')
    parser.add_argument('--config',      type=str,   help='Path to YAML config file')
    parser.add_argument('--encoder',     type=str,   choices=list(ENCODER_CONFIGS),
                        help='Encoder backbone to use')
    parser.add_argument('--mode',        type=str,   choices=['search', 'full'],
                        help='Dataset mode: search (subset) or full')
    parser.add_argument('--epochs',      type=int)
    parser.add_argument('--batch_size',  type=int)
    parser.add_argument('--lr',          type=float)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--output_dir',  type=str,
                        help='Directory to save best model checkpoint')
    parser.add_argument('--project',     type=str,
                        help='W&B project name')
    
    args = parser.parse_args()
    
    # Load YAML if provided
    cfg = {}
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    
    # Merge logic (CLI args override YAML)
    # We flatten or map the nested YAML to the args namespace for simplicity
    final_cfg = argparse.Namespace()
    
    # Default values or YAML values
    final_cfg.encoder = args.encoder or cfg.get('model', {}).get('encoder', 'resnet18')
    final_cfg.mode = args.mode or cfg.get('data', {}).get('mode', 'search')
    final_cfg.epochs = args.epochs or cfg.get('training', {}).get('epochs', 5)
    final_cfg.batch_size = args.batch_size or cfg.get('training', {}).get('batch_size', 32)
    final_cfg.lr = args.lr or cfg.get('training', {}).get('lr', 1e-3)
    final_cfg.num_workers = args.num_workers or cfg.get('training', {}).get('num_workers', 4)
    final_cfg.output_dir = args.output_dir or cfg.get('output_dir', 'results')
    final_cfg.project = args.project or cfg.get('project', 'C5-ImageCaptioning')
    final_cfg.scheduler = cfg.get('training', {}).get('scheduler', 'plateau')
    
    return final_cfg

def convert_indices_to_string(indices):
    res = ""
    for idx in indices:
        char = idx2char[idx.item()]
        if char == '<EOS>' or char == '<PAD>':
            break
        if char != '<SOS>':
            res += char
    return res

def train_one_epoch(model, optimizer, crit, dataloader, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}", mininterval=30.0)
    
    for img, caption, _ in progress_bar:
        img, caption = img.to(DEVICE), caption.to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(img, caption)
        # caption[:, 1:] are the tokens we want to predict (skipping <SOS>)
        loss = crit(pred, caption[:, 1:])
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, crit):
    model.eval()
    all_preds = []
    all_refs = []
    all_refs_old = []
    total_images = 0
    total_inference_time = 0.0  # seconds, model forward only
    batch_latencies = []         # per-batch forward times (seconds)
    total_val_loss = 0.0
    val_loss_steps = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("Evaluating...")
    eval_start = time.perf_counter()

    sample_indices = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450] 
    samples_to_print = []

    with torch.no_grad():
        for i, (img, caption, img_names) in enumerate(tqdm(dataloader, desc="Eval", mininterval=30.0)):
            img, caption = img.to(DEVICE), caption.to(DEVICE)

            # --- Calculate Validation Loss (Teacher Forcing) ---
            # We use teacher forcing during eval loss to see how well the model "knows" the sequences
            val_pred = model(img, caption)
            v_loss = crit(val_pred, caption[:, 1:])
            total_val_loss += v_loss.item()
            val_loss_steps += 1

            # --- time the forward pass only (Auto-regressive) ---
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = model(img) # Generative pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_latencies.append(t1 - t0)
            total_inference_time += t1 - t0
            total_images += img.size(0)

            pred_indices = pred.argmax(dim=1)
            for b in range(img.size(0)):
                pred_str = convert_indices_to_string(pred_indices[b])
                # Retrieve all valid references for this image from the dataset
                actual_img_id = dataloader.dataset.valid_image_ids[i * dataloader.batch_size + b]
                ref_strs = dataloader.dataset.image_captions[actual_img_id]
                
                all_preds.append(pred_str)
                all_refs.append(ref_strs)
                all_refs_old.append([ref_strs[0]])
                
                # Collect samples (store all references)
                global_idx = i * dataloader.batch_size + b
                if global_idx in sample_indices:
                    samples_to_print.append((pred_str, ref_strs, img_names[b]))

    eval_end = time.perf_counter()
    total_run_latency = eval_end - eval_start

    # --- compute metrics ---
    try:
        bleu1 = bleu.compute(predictions=all_preds, references=all_refs, max_order=1)
        bleu2 = bleu.compute(predictions=all_preds, references=all_refs, max_order=2)
        res_r = rouge.compute(predictions=all_preds, references=all_refs)
        res_m = meteor.compute(predictions=all_preds, references=all_refs)
        
        bleu1_old = bleu.compute(predictions=all_preds, references=all_refs_old, max_order=1)
        bleu2_old = bleu.compute(predictions=all_preds, references=all_refs_old, max_order=2)
        res_r_old = rouge.compute(predictions=all_preds, references=all_refs_old)
        res_m_old = meteor.compute(predictions=all_preds, references=all_refs_old)

        cider_score = 0.0
        cider_score_old = 0.0
        if CIDER_AVAILABLE:
            cider_score = cider.compute(predictions=all_preds, references=all_refs)['cider'] * 100
            cider_score_old = cider.compute(predictions=all_preds, references=all_refs_old)['cider'] * 100

        metrics = {
            "BLEU-1":  bleu1['bleu'] * 100,
            "BLEU-2":  bleu2['bleu'] * 100,
            "ROUGE-L": res_r['rougeL'] * 100,
            "METEOR":  res_m['meteor'] * 100,
            "CIDEr":   cider_score,
            "BLEU-1_old":  bleu1_old['bleu'] * 100,
            "BLEU-2_old":  bleu2_old['bleu'] * 100,
            "ROUGE-L_old": res_r_old['rougeL'] * 100,
            "METEOR_old":  res_m_old['meteor'] * 100,
            "CIDEr_old":   cider_score_old,
        }
    except Exception as e:
        print(f"Failed computing metrics (possibly empty predictions): {e}")
        metrics = {"BLEU-1": 0, "BLEU-2": 0, "ROUGE-L": 0, "METEOR": 0, "CIDEr": 0,
                   "BLEU-1_old": 0, "BLEU-2_old": 0, "ROUGE-L_old": 0, "METEOR_old": 0, "CIDEr_old": 0}

    # --- Print Samples ---
    print("\n--- Evaluation Samples ---")
    for s_pred, s_refs, s_img in samples_to_print:
        print(f"  Img:  {s_img}")
        for idx, r in enumerate(s_refs):
            print(f"  Ref {idx+1}: {r}")
        print(f"  Pred:  {s_pred}")
        print("-" * 20)

    # --- compute metrics ---
    fps = total_images / total_inference_time if total_inference_time > 0 else 0.0
    avg_latency_ms = (total_inference_time / len(batch_latencies) * 1000) if batch_latencies else 0.0
    max_vram_gb = (torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0.0

    compute_metrics = {
        "val_loss":                      total_val_loss / val_loss_steps if val_loss_steps > 0 else 0.0,
        "compute/total_run_latency_s":   round(total_run_latency, 3),
        "compute/total_inference_time_s": round(total_inference_time, 3),
        "compute/fps":                   round(fps, 2),
        "compute/avg_batch_latency_ms":  round(avg_latency_ms, 3),
        "compute/max_vram_gb":           round(max_vram_gb, 4),
    }
    metrics.update(compute_metrics)
    return metrics, samples_to_print

def main():
    args = parse_args()

    run = wandb.init(
        project=args.project,
        config=vars(args),
        name=f"{args.encoder}_lr{args.lr}_bs{args.batch_size}",
    )
    cfg = wandb.config  # sweep may override args values

    print(f"Config: encoder={cfg.encoder}, mode={cfg.mode}, lr={cfg.lr}, batch_size={cfg.batch_size}, epochs={cfg.epochs}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    run_output_dir = os.path.join(cfg.output_dir, f"{run.name}_{run.id}")
    os.makedirs(run_output_dir, exist_ok=True)
    ckpt_path = os.path.join(run_output_dir, "best_model.pt")

    # Dir for visual samples
    samples_dir = os.path.join(run_output_dir, "visual_samples")
    os.makedirs(samples_dir, exist_ok=True)
    history_file = os.path.join(run_output_dir, "captions_history.json")
    caption_history = {} # epoch -> samples

    # Save run configuration
    with open(os.path.join(run_output_dir, "config.json"), "w") as f:
        json.dump(dict(cfg), f, indent=4)

    print("Loading datasets...")
    if cfg.mode == "search":
        # In search mode, val data is also from TRAIN_ANN (10% split)
        dataset_train = VizWizDataset(TRAIN_ANN, TRAIN_IMG_DIR, split="train_search", mode=cfg.mode)
        dataset_valid = VizWizDataset(TRAIN_ANN, TRAIN_IMG_DIR, split="val_search", mode=cfg.mode)
    else:
        dataset_train = VizWizDataset(TRAIN_ANN, TRAIN_IMG_DIR, split="train", mode=cfg.mode)
        dataset_valid = VizWizDataset(VAL_ANN, VAL_IMG_DIR, split="val", mode=cfg.mode)

    dataloader_train = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers)

    print(f"Initializing model with encoder: {cfg.encoder} ...")
    model = ImageCaptioningModel(encoder_name=cfg.encoder).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    scheduler = None
    if cfg.scheduler == "plateau":
        print("Using ReduceLROnPlateau scheduler")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # ignore_index=2 corresponds to <PAD> token
    crit = nn.CrossEntropyLoss(ignore_index=2)

    # METEOR is the primary metric (best correlation with human judgement among available).
    # CIDEr is also logged. BLEU/ROUGE are secondary.
    primary_metric_name = "METEOR"
    best_primary = -1.0

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, optimizer, crit, dataloader_train, epoch)
        print(f"End of Epoch {epoch} - Average Train Loss: {loss:.4f}")

        metrics, eval_samples = eval_epoch(model, dataloader_valid, crit)
        
        # Save samples evolution
        epoch_samples = []
        for pred, refs, img_name in eval_samples:
            epoch_samples.append({
                "image_name": img_name,
                "references": refs,
                "prediction": pred
            })
            # On first epoch, copy the images to results for easy access
            if epoch == 1:
                src_img = os.path.join(TRAIN_IMG_DIR if cfg.mode == "search" else VAL_IMG_DIR, img_name)
                if os.path.exists(src_img):
                    shutil.copy(src_img, os.path.join(samples_dir, img_name))
        
        caption_history[epoch] = epoch_samples
        with open(history_file, "w") as f:
            json.dump(caption_history, f, indent=4)

        print("Validation Metrics:")
        print(f"  [New] BLEU-1: {metrics.get('BLEU-1', 0):.2f}% | BLEU-2: {metrics.get('BLEU-2', 0):.2f}% | "
              f"ROUGE-L: {metrics.get('ROUGE-L', 0):.2f}% | METEOR: {metrics.get('METEOR', 0):.4f}% | "
              f"CIDEr: {metrics.get('CIDEr', 0):.2f}")
        print(f"  [Old] BLEU-1: {metrics.get('BLEU-1_old', 0):.2f}% | BLEU-2: {metrics.get('BLEU-2_old', 0):.2f}% | "
              f"ROUGE-L: {metrics.get('ROUGE-L_old', 0):.2f}% | METEOR: {metrics.get('METEOR_old', 0):.4f}% | "
              f"CIDEr: {metrics.get('CIDEr_old', 0):.2f}")
        print(f"  Val Loss: {metrics.get('val_loss', 0):.4f}")
        print(f"  FPS: {metrics.get('compute/fps', 0):.1f} | "
              f"Avg batch latency: {metrics.get('compute/avg_batch_latency_ms', 0):.1f}ms | "
              f"Inference time: {metrics.get('compute/total_inference_time_s', 0):.1f}s | "
              f"Run latency: {metrics.get('compute/total_run_latency_s', 0):.1f}s | "
              f"Max VRAM: {metrics.get('compute/max_vram_gb', 0):.3f}GB")

        wandb.log({"epoch": epoch, "train_loss": loss, "lr": optimizer.param_groups[0]['lr'], **metrics})

        # Step scheduler based on METEOR (primary metric)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics.get(primary_metric_name, 0))
            else:
                scheduler.step()

        if metrics.get(primary_metric_name, 0) > best_primary:
            best_primary = metrics[primary_metric_name]
            torch.save(model.state_dict(), ckpt_path)
            
            # Save best metrics
            with open(os.path.join(run_output_dir, "best_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            print(f"  -> New best {primary_metric_name}: {best_primary:.2f} — checkpoint saved to {ckpt_path}")
            wandb.run.summary[f"best_{primary_metric_name}"] = best_primary
            # also log compute metrics as run summary on first best
            for k, v in metrics.items():
                if k.startswith("compute/"):
                    wandb.run.summary[k] = v

    wandb.finish()


if __name__ == "__main__":
    main()
