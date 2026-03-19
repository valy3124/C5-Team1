import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import time
import json
import argparse
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import wandb
import shutil
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from dataset import VizWizDataset, CharTokenizer, SubwordTokenizer, WordTokenizer
from model import ImageCaptioningModel, ENCODER_CONFIGS

# --- Paths ---
DATA_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'VizWiz')
TRAIN_ANN     = os.path.join(DATA_BASE_DIR, 'annotations', 'train.json')
VAL_ANN       = os.path.join(DATA_BASE_DIR, 'annotations', 'val.json')
TRAIN_IMG_DIR = os.path.join(DATA_BASE_DIR, 'images', 'train')
VAL_IMG_DIR   = os.path.join(DATA_BASE_DIR, 'images', 'val')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

bleu   = evaluate.load('bleu')
rouge  = evaluate.load('rouge')
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
    parser.add_argument('--config',      type=str, help='Path to YAML config file')
    parser.add_argument('--encoder',     type=str, choices=list(ENCODER_CONFIGS))
    parser.add_argument('--mode',        type=str, choices=['search', 'full'],
                        help='Dataset mode: search (subset) or full')
    parser.add_argument('--epochs',      type=int)
    parser.add_argument('--batch_size',  type=int)
    parser.add_argument('--lr',          type=float)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--output_dir',     type=str)
    parser.add_argument('--project',         type=str,   help='W&B project name')
    parser.add_argument('--freeze_encoder',  type=lambda x: x.lower() != 'false',
                        default=None,        help='Freeze encoder weights (true/false)')
    parser.add_argument('--grad_clip',       type=float, default=None,
                        help='Max gradient norm for clipping (0 = disabled)')
    parser.add_argument('--decoder_type',   type=str, choices=['gru', 'lstm', 'xlstm'])
    parser.add_argument('--decoder_dim',    type=int)
    parser.add_argument('--decoder_layers', type=int)
    parser.add_argument('--embed_dim',      type=int)
    parser.add_argument('--text_level',     type=str, choices=['char', 'subword', 'word'],
                        help='Text representation level')
    parser.add_argument('--clip_embeddings', action='store_true', default=None, help='Initialize embeddings with CLIP pretrained weights')
    parser.add_argument('--freeze_embeddings', action='store_true', default=None, help='Freeze the token embeddings weights')

    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

    # CLI args take priority over YAML; use 'is not None' so numeric 0-values are respected.
    final_cfg = argparse.Namespace()
    final_cfg.encoder      = args.encoder     if args.encoder     is not None else cfg.get('model', {}).get('encoder', 'resnet18')
    final_cfg.mode         = args.mode        if args.mode        is not None else cfg.get('data', {}).get('mode', 'search')
    final_cfg.epochs       = args.epochs      if args.epochs      is not None else cfg.get('training', {}).get('epochs', 5)
    final_cfg.batch_size   = args.batch_size  if args.batch_size  is not None else cfg.get('training', {}).get('batch_size', 32)
    final_cfg.lr           = args.lr          if args.lr          is not None else cfg.get('training', {}).get('lr', 1e-3)
    final_cfg.num_workers  = args.num_workers if args.num_workers is not None else cfg.get('training', {}).get('num_workers', 4)
    final_cfg.output_dir   = args.output_dir  if args.output_dir  is not None else cfg.get('output_dir', 'results')
    final_cfg.project      = args.project     if args.project     is not None else cfg.get('project', 'C5-ImageCaptioning')
    final_cfg.scheduler    = cfg.get('training', {}).get('scheduler', 'plateau')
    final_cfg.grad_clip    = args.grad_clip    if args.grad_clip    is not None else cfg.get('training', {}).get('grad_clip', 5.0)
    final_cfg.freeze_encoder = args.freeze_encoder if args.freeze_encoder is not None else cfg.get('model', {}).get('freeze_encoder', False)
    
    final_cfg.decoder_type   = args.decoder_type   if args.decoder_type   is not None else cfg.get('model', {}).get('decoder_type', 'gru')
    final_cfg.decoder_dim    = args.decoder_dim    if args.decoder_dim    is not None else cfg.get('model', {}).get('decoder_dim', 512)
    final_cfg.decoder_layers = args.decoder_layers if args.decoder_layers is not None else cfg.get('model', {}).get('decoder_layers', 1)
    final_cfg.embed_dim      = args.embed_dim      if args.embed_dim      is not None else cfg.get('model', {}).get('embed_dim', 512)
    final_cfg.text_level     = args.text_level     if args.text_level     is not None else cfg.get('model', {}).get('text_level', 'char')
    final_cfg.clip_embeddings = args.clip_embeddings if args.clip_embeddings is not None else cfg.get('model', {}).get('clip_embeddings', False)
    final_cfg.freeze_embeddings = args.freeze_embeddings if args.freeze_embeddings is not None else cfg.get('model', {}).get('freeze_embeddings', False)
    
    return final_cfg


def build_word_vocab(annotation_file, min_freq=5):
    import json
    from collections import Counter
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    counter = Counter()
    for ann in data['annotations']:
        if ann.get('is_precanned', False) or ann.get('is_rejected', False):
            continue
        caption = ann['caption'].lower()
        tokens = word_tokenize(caption)
        counter.update(tokens)
    
    # Filter by frequency
    words = [word for word, count in counter.items() if count >= min_freq]
    
    vocab = ["<SOS>", "<EOS>", "<PAD>", "<UNK>"] + sorted(words)
    return vocab


def convert_indices_to_string(indices, tokenizer):
    return tokenizer.decode(indices)


def train_one_epoch(model, optimizer, crit, dataloader, epoch, grad_clip=5.0):
    model.train()
    total_loss = 0
    grad_norms = []
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}", mininterval=30.0)

    for img, caption, _ in progress_bar:
        img, caption = img.to(DEVICE), caption.to(DEVICE)

        optimizer.zero_grad()
        pred = model(img, caption)
        loss = crit(pred, caption[:, 1:])  # predict caption[:, 1:] (skip <SOS>)

        loss.backward()
        raw_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        grad_norms.append(raw_norm.item())
        optimizer.step()

        total_loss += loss.item()

    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    max_grad_norm = max(grad_norms) if grad_norms else 0
    return total_loss / len(dataloader), avg_grad_norm, max_grad_norm


def eval_epoch(model, dataloader, crit, tokenizer):
    model.eval()
    all_preds, all_refs, all_refs_old = [], [], []
    total_images         = 0
    total_inference_time = 0.0
    batch_latencies      = []
    total_val_loss       = 0.0
    val_loss_steps       = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("Evaluating...")
    eval_start = time.perf_counter()

    filename_to_img_id = {v: k for k, v in dataloader.dataset.images.items()}

    sample_indices  = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    samples_to_print = []

    with torch.no_grad():
        for i, (img, caption, img_names) in enumerate(tqdm(dataloader, desc="Eval", mininterval=30.0)):
            img, caption = img.to(DEVICE), caption.to(DEVICE)

            # Validation loss via teacher forcing
            val_pred = model(img, caption)
            v_loss   = crit(val_pred, caption[:, 1:])
            total_val_loss += v_loss.item()
            val_loss_steps += 1

            # Auto-regressive inference (timed)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0   = time.perf_counter()
            pred = model(img)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1   = time.perf_counter()

            batch_latencies.append(t1 - t0)
            total_inference_time += t1 - t0
            total_images += img.size(0)

            pred_indices = pred.argmax(dim=1)
            for b in range(img.size(0)):
                pred_str      = convert_indices_to_string(pred_indices[b], tokenizer)
                actual_img_id = filename_to_img_id[img_names[b]]
                ref_strs      = dataloader.dataset.image_captions[actual_img_id]

                all_preds.append(pred_str)
                all_refs.append(ref_strs)
                all_refs_old.append([ref_strs[0]])

                global_idx = i * dataloader.batch_size + b
                if global_idx in sample_indices:
                    samples_to_print.append((pred_str, ref_strs, img_names[b]))

    eval_end          = time.perf_counter()
    total_run_latency = eval_end - eval_start

    try:
        bleu1 = bleu.compute(predictions=all_preds, references=all_refs, max_order=1)
        bleu2 = bleu.compute(predictions=all_preds, references=all_refs, max_order=2)
        res_r = rouge.compute(predictions=all_preds, references=all_refs)
        res_m = meteor.compute(predictions=all_preds, references=all_refs)

        bleu1_old = bleu.compute(predictions=all_preds, references=all_refs_old, max_order=1)
        bleu2_old = bleu.compute(predictions=all_preds, references=all_refs_old, max_order=2)
        res_r_old = rouge.compute(predictions=all_preds, references=all_refs_old)
        res_m_old = meteor.compute(predictions=all_preds, references=all_refs_old)

        cider_score = cider_score_old = 0.0
        if CIDER_AVAILABLE:
            cider_score     = cider.compute(predictions=all_preds, references=all_refs)['cider'] * 100
            cider_score_old = cider.compute(predictions=all_preds, references=all_refs_old)['cider'] * 100

        metrics = {
            "BLEU-1":  bleu1['bleu'] * 100 if bleu1 else 0.0,
            "BLEU-2":  bleu2['bleu'] * 100 if bleu2 else 0.0,
            "ROUGE-L": res_r['rougeL'] * 100 if res_r else 0.0,
            "METEOR":  res_m['meteor'] * 100 if res_m else 0.0,
            "CIDEr":   cider_score,
            "BLEU-1_old":  bleu1_old['bleu'] * 100 if bleu1_old else 0.0,
            "BLEU-2_old":  bleu2_old['bleu'] * 100 if bleu2_old else 0.0,
            "ROUGE-L_old": res_r_old['rougeL'] * 100 if res_r_old else 0.0,
            "METEOR_old":  res_m_old['meteor'] * 100 if res_m_old else 0.0,
            "CIDEr_old":   cider_score_old,
        }
    except Exception as e:
        print(f"Failed computing metrics: {e}")
        metrics = {k: 0 for k in ["BLEU-1", "BLEU-2", "ROUGE-L", "METEOR", "CIDEr",
                                   "BLEU-1_old", "BLEU-2_old", "ROUGE-L_old", "METEOR_old", "CIDEr_old"]}

    print("\n--- Evaluation Samples ---")
    for s_pred, s_refs, s_img in samples_to_print:
        print(f"  Img:  {s_img}")
        for idx, r in enumerate(s_refs):
            print(f"  Ref {idx+1}: {r}")
        print(f"  Pred:  {s_pred}")
        print("-" * 20)

    fps            = total_images / total_inference_time if total_inference_time > 0 else 0.0
    avg_latency_ms = (total_inference_time / len(batch_latencies) * 1000) if batch_latencies else 0.0
    max_vram_gb    = (torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0.0

    metrics.update({
        "val_loss":                       total_val_loss / val_loss_steps if val_loss_steps > 0 else 0.0,
        "compute/total_run_latency_s":    round(total_run_latency, 3),
        "compute/total_inference_time_s": round(total_inference_time, 3),
        "compute/fps":                    round(fps, 2),
        "compute/avg_batch_latency_ms":   round(avg_latency_ms, 3),
        "compute/max_vram_gb":            round(max_vram_gb, 4),
    })
    return metrics, samples_to_print


def main():
    final_cfg = parse_args()
    cfg = final_cfg # renaming for clarity in the rest of main

    run_name = f"{cfg.encoder}_{cfg.decoder_type}_d{cfg.decoder_dim}_l{cfg.decoder_layers}_{cfg.text_level}"
    if cfg.clip_embeddings:
        suffix = "_frozen" if cfg.freeze_embeddings else "_unfrozen"
        run_name += f"_clip_embed{suffix}"
        
    run = wandb.init(
        project=cfg.project,
        config=vars(cfg),
        name=run_name,
    )
    cfg = wandb.config  # W&B sweep may override values

    print(f"Config: encoder={cfg.encoder}, text_level={cfg.text_level}, mode={cfg.mode}, lr={cfg.lr}, "
          f"batch_size={cfg.batch_size}, epochs={cfg.epochs}, clip_embed={cfg.clip_embeddings}, frozen={cfg.freeze_embeddings}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    run_output_dir = os.path.join(cfg.output_dir, f"{run.name}_{run.id}")
    os.makedirs(run_output_dir, exist_ok=True)
    ckpt_path    = os.path.join(run_output_dir, "best_model.pt")
    samples_dir  = os.path.join(run_output_dir, "visual_samples")
    history_file = os.path.join(run_output_dir, "captions_history.json")
    os.makedirs(samples_dir, exist_ok=True)
    caption_history = {}

    with open(os.path.join(run_output_dir, "config.json"), "w") as f:
        json.dump(dict(cfg), f, indent=4)

    # Initialize Tokenizer
    print(f"Initializing {cfg.text_level} tokenizer...")
    if cfg.text_level == 'char':
        tokenizer = CharTokenizer()
    elif cfg.text_level == 'subword':
        # Use CLIP tokenizer
        tokenizer = SubwordTokenizer()
    elif cfg.text_level == 'word':
        vocab = build_word_vocab(TRAIN_ANN)
        tokenizer = WordTokenizer(vocab=vocab)
    else:
        raise ValueError(f"Unknown text_level: {cfg.text_level}")

    print("Loading datasets...")
    # Auto-select correct image normalization for the chosen encoder
    encoder_source = ENCODER_CONFIGS.get(cfg.encoder, ('hf',))[0]
    if encoder_source == 'clip':
        img_mean = VizWizDataset.CLIP_MEAN
        img_std  = VizWizDataset.CLIP_STD
    else:
        img_mean = VizWizDataset.IMAGENET_MEAN
        img_std  = VizWizDataset.IMAGENET_STD

    if cfg.mode == "search":
        dataset_train = VizWizDataset(TRAIN_ANN, TRAIN_IMG_DIR, split="train_search", mode=cfg.mode,
                                      img_mean=img_mean, img_std=img_std, tokenizer=tokenizer)
        dataset_valid = VizWizDataset(TRAIN_ANN, TRAIN_IMG_DIR, split="val_search",   mode=cfg.mode,
                                      img_mean=img_mean, img_std=img_std, tokenizer=tokenizer)
    else:
        dataset_train = VizWizDataset(TRAIN_ANN, TRAIN_IMG_DIR, split="train", mode=cfg.mode,
                                      img_mean=img_mean, img_std=img_std, tokenizer=tokenizer)
        dataset_valid = VizWizDataset(VAL_ANN,   VAL_IMG_DIR,   split="val",   mode=cfg.mode,
                                      img_mean=img_mean, img_std=img_std, tokenizer=tokenizer)

    dataloader_train = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers)

    print(f"Initializing model with encoder={cfg.encoder}, freeze_encoder={cfg.freeze_encoder} ...")
    model = ImageCaptioningModel(
        encoder_name=cfg.encoder,
        freeze_encoder=cfg.freeze_encoder,
        decoder_type=cfg.decoder_type,
        decoder_dim=cfg.decoder_dim,
        decoder_layers=cfg.decoder_layers,
        embed_dim=cfg.embed_dim,
        vocab_size=tokenizer.vocab_size,
        sos_idx=tokenizer.sos_idx,
        eos_idx=tokenizer.eos_idx,
        pad_idx=tokenizer.pad_idx,
        max_len=tokenizer.max_len,
        clip_embeddings=cfg.clip_embeddings,
        clip_model_id='openai/clip-vit-base-patch32' if cfg.text_level == 'subword' else None,
        freeze_embeddings=cfg.freeze_embeddings
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"trainable_params": trainable_params, "vocab_size": tokenizer.vocab_size}, allow_val_change=True)

    scheduler = None
    if cfg.scheduler == "plateau":
        print("Using ReduceLROnPlateau scheduler")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    primary_metric_name = "METEOR"
    best_primary = -1.0

    for epoch in range(1, cfg.epochs + 1):
        loss, avg_grad_norm, max_grad_norm = train_one_epoch(
            model, optimizer, crit, dataloader_train, epoch, grad_clip=cfg.grad_clip
        )
        print(f"End of Epoch {epoch} - Train Loss: {loss:.4f} | "
              f"Grad Norm (avg/max before clip): {avg_grad_norm:.3f} / {max_grad_norm:.3f}")

        metrics, eval_samples = eval_epoch(model, dataloader_valid, crit, tokenizer)

        epoch_samples = []
        for pred, refs, img_name in eval_samples:
            epoch_samples.append({"image_name": img_name, "references": refs, "prediction": pred})
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

        wandb.log({
            "epoch":         epoch,
            "train_loss":    loss,
            "lr":            optimizer.param_groups[0]['lr'],
            "grad_norm/avg": avg_grad_norm,
            "grad_norm/max": max_grad_norm,
            "best_METEOR":   best_primary,
            **metrics,
        })

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics.get(primary_metric_name, 0))
            else:
                scheduler.step()

        if metrics.get(primary_metric_name, 0) > best_primary:
            best_primary = metrics[primary_metric_name]
            torch.save(model.state_dict(), ckpt_path)
            with open(os.path.join(run_output_dir, "best_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"  -> New best {primary_metric_name}: {best_primary:.2f} — checkpoint saved to {ckpt_path}")
            wandb.run.summary[f"best_{primary_metric_name}"] = best_primary
            for k, v in metrics.items():
                if k.startswith("compute/"):
                    wandb.run.summary[k] = v

    wandb.finish()


if __name__ == "__main__":
    main()

