import os
import time
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import wandb

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


def parse_args():
    parser = argparse.ArgumentParser(description='Train image captioning model on VizWiz')
    parser.add_argument('--encoder',     type=str,   default='resnet18', choices=list(ENCODER_CONFIGS),
                        help='Encoder backbone to use')
    parser.add_argument('--mode',        type=str,   default='search', choices=['search', 'full'],
                        help='Dataset mode: search (subset) or full')
    parser.add_argument('--epochs',      type=int,   default=5)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--output_dir',  type=str,   default='results/checkpoints',
                        help='Directory to save best model checkpoint')
    parser.add_argument('--project',     type=str,   default='C5-ImageCaptioning',
                        help='W&B project name')
    return parser.parse_args()

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
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}", mininterval=10.0)
    
    for img, caption in progress_bar:
        img, caption = img.to(DEVICE), caption.to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(img)
        loss = crit(pred, caption)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader):
    model.eval()
    all_preds = []
    all_refs = []
    total_images = 0
    total_inference_time = 0.0  # seconds, model forward only
    batch_latencies = []         # per-batch forward times (seconds)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("Evaluating...")
    eval_start = time.perf_counter()

    with torch.no_grad():
        for img, caption in tqdm(dataloader, desc="Eval", mininterval=10.0):
            img = img.to(DEVICE)

            # --- time the forward pass only ---
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = model(img)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_latencies.append(t1 - t0)
            total_inference_time += t1 - t0
            total_images += img.size(0)

            pred_indices = pred.argmax(dim=1)
            for b in range(img.size(0)):
                pred_str = convert_indices_to_string(pred_indices[b])
                ref_str = convert_indices_to_string(caption[b])
                all_preds.append(pred_str)
                all_refs.append([ref_str])

    eval_end = time.perf_counter()
    total_run_latency = eval_end - eval_start

    # --- compute metrics ---
    try:
        bleu1 = bleu.compute(predictions=all_preds, references=all_refs, max_order=1)
        bleu2 = bleu.compute(predictions=all_preds, references=all_refs, max_order=2)
        res_r = rouge.compute(predictions=all_preds, references=all_refs)
        res_m = meteor.compute(predictions=all_preds, references=all_refs)
        cider_score = 0.0
        if CIDER_AVAILABLE:
            cider_score = cider.compute(predictions=all_preds, references=all_refs)['cider'] * 100

        metrics = {
            "BLEU-1":  bleu1['bleu'] * 100,
            "BLEU-2":  bleu2['bleu'] * 100,
            "ROUGE-L": res_r['rougeL'] * 100,
            "METEOR":  res_m['meteor'] * 100,
            "CIDEr":   cider_score,
        }
    except Exception as e:
        print(f"Failed computing metrics (possibly empty predictions): {e}")
        metrics = {"BLEU-1": 0, "BLEU-2": 0, "ROUGE-L": 0, "METEOR": 0, "CIDEr": 0}

    # --- compute metrics ---
    fps = total_images / total_inference_time if total_inference_time > 0 else 0.0
    avg_latency_ms = (total_inference_time / len(batch_latencies) * 1000) if batch_latencies else 0.0
    max_vram_gb = (torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0.0

    compute_metrics = {
        "compute/total_run_latency_s":   round(total_run_latency, 3),
        "compute/total_inference_time_s": round(total_inference_time, 3),
        "compute/fps":                   round(fps, 2),
        "compute/avg_batch_latency_ms":  round(avg_latency_ms, 3),
        "compute/max_vram_gb":           round(max_vram_gb, 4),
    }
    metrics.update(compute_metrics)
    return metrics

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
    ckpt_path = os.path.join(cfg.output_dir, f"{run.name}.pt")

    print("Loading datasets...")
    dataset_train = VizWizDataset(TRAIN_ANN, TRAIN_IMG_DIR, split="train", mode=cfg.mode)
    dataset_valid = VizWizDataset(VAL_ANN, VAL_IMG_DIR, split="val", mode=cfg.mode)

    dataloader_train = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers)

    print(f"Initializing model with encoder: {cfg.encoder} ...")
    model = ImageCaptioningModel(encoder_name=cfg.encoder).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()

    # METEOR is the primary metric (best correlation with human judgement among available).
    # CIDEr is also logged. BLEU/ROUGE are secondary.
    primary_metric_name = "METEOR"
    best_primary = -1.0

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, optimizer, crit, dataloader_train, epoch)
        print(f"End of Epoch {epoch} - Average Train Loss: {loss:.4f}")

        metrics = eval_epoch(model, dataloader_valid)
        print("Validation Metrics:")
        print(f"  BLEU-1: {metrics.get('BLEU-1', 0):.2f}% | BLEU-2: {metrics.get('BLEU-2', 0):.2f}% | "
              f"ROUGE-L: {metrics.get('ROUGE-L', 0):.2f}% | METEOR: {metrics.get('METEOR', 0):.4f} | "
              f"CIDEr: {metrics.get('CIDEr', 0):.2f}")
        print(f"  FPS: {metrics.get('compute/fps', 0):.1f} | "
              f"Avg batch latency: {metrics.get('compute/avg_batch_latency_ms', 0):.1f}ms | "
              f"Inference time: {metrics.get('compute/total_inference_time_s', 0):.1f}s | "
              f"Run latency: {metrics.get('compute/total_run_latency_s', 0):.1f}s | "
              f"Max VRAM: {metrics.get('compute/max_vram_gb', 0):.3f}GB")

        wandb.log({"epoch": epoch, "train_loss": loss, **metrics})

        if metrics.get(primary_metric_name, 0) > best_primary:
            best_primary = metrics[primary_metric_name]
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best {primary_metric_name}: {best_primary:.2f} — checkpoint saved to {ckpt_path}")
            wandb.run.summary[f"best_{primary_metric_name}"] = best_primary
            # also log compute metrics as run summary on first best
            for k, v in metrics.items():
                if k.startswith("compute/"):
                    wandb.run.summary[k] = v

    wandb.finish()


if __name__ == "__main__":
    main()
