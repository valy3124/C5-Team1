import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import math
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


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

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
    parser.add_argument('--grad_clip',       type=float, default=None)
    parser.add_argument('--decoder_type',   type=str, choices=['gru', 'lstm', 'xlstm'])
    parser.add_argument('--decoder_dim',    type=int)
    parser.add_argument('--decoder_layers', type=int)
    parser.add_argument('--embed_dim',      type=int)
    parser.add_argument('--text_level',     type=str, choices=['char', 'subword', 'word'])
    parser.add_argument('--clip_embeddings',   action='store_true', default=None)
    parser.add_argument('--freeze_embeddings', action='store_true', default=None)
    # --- Attention ---
    parser.add_argument('--attn_type', type=str, choices=['soft', 'adaptive', 'early_fusion'], default=None,
                        help='Type of visual attention to use. "soft", "adaptive", or "early_fusion" (recommended for xLSTM)')
    parser.add_argument('--attn_dim', type=int, default=None,
                        help='Hidden dimension of the attention scoring network (default 256)')

    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

    # CLI args take priority over YAML
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
    final_cfg.clip_embeddings  = args.clip_embeddings  if args.clip_embeddings  is not None else cfg.get('model', {}).get('clip_embeddings', False)
    final_cfg.freeze_embeddings = args.freeze_embeddings if args.freeze_embeddings is not None else cfg.get('model', {}).get('freeze_embeddings', False)
    final_cfg.attn_type     = args.attn_type if args.attn_type is not None else cfg.get('model', {}).get('attn_type', None)
    final_cfg.attn_dim      = args.attn_dim      if args.attn_dim      is not None else cfg.get('model', {}).get('attn_dim', 256)
    
    # Retrocompatibility with old configs
    if final_cfg.attn_type is None and cfg.get('model', {}).get('use_attention', False):
        final_cfg.attn_type = 'soft'

    return final_cfg


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

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
        tokens = word_tokenize(ann['caption'].lower())
        counter.update(tokens)

    words = [w for w, c in counter.items() if c >= min_freq]
    return ["<SOS>", "<EOS>", "<PAD>", "<UNK>"] + sorted(words)


def convert_indices_to_string(indices, tokenizer):
    return tokenizer.decode(indices)


# ---------------------------------------------------------------------------
#  Attention visualization
# ---------------------------------------------------------------------------

def generate_attention_visualizations(
    model, tokenizer, sample_data, epoch, run_output_dir, img_dir
):
    """
    For each sample image, generate per-word attention heatmap images.

    Args:
        sample_data : list of (pred_str, ref_strs, img_name, img_tensor)
        epoch       : current epoch number (used in folder name)
        run_output_dir : base output dir for this run
        img_dir     : directory where the original images live
    Output structure:
        {run_output_dir}/attention_outputs_epoch_{epoch:02d}/
            {img_stem}_outputs/
                word_00_{tok}.png   # one per generated token
                attention_grid.png  # composite grid (paper-style)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image as PILImage

    attn_dir = os.path.join(run_output_dir, f"attention_outputs_epoch_{epoch:02d}")
    os.makedirs(attn_dir, exist_ok=True)
    print(f"\n[Attention Viz] Generating visualizations → {attn_dir}")

    model.eval()
    with torch.no_grad():
        for pred_str, ref_strs, img_name, img_tensor in sample_data:
            img_input = img_tensor.unsqueeze(0).to(DEVICE)     # (1, C, H, W)
            orig_img  = PILImage.open(os.path.join(img_dir, img_name)).convert('RGB')
            orig_arr  = np.array(orig_img)
            
            # --- early_fusion (xLSTM) Plotting Route ---
            if getattr(model, 'attn_type', None) == 'early_fusion':
                img_stem    = os.path.splitext(img_name)[0]
                img_out_dir = os.path.join(attn_dir, f"{img_stem}_outputs")
                os.makedirs(img_out_dir, exist_ok=True)
                try:
                    from xlstm_plotter import plot_xlstm_visualizations
                    result, pseudo_attn_maps, surprise_map = model.generate_with_pseudo_attention_and_surprise(img_input)
                    tokens = [t for t in result.argmax(dim=1)[0].tolist() if t != getattr(tokenizer, 'eos_idx', -1)]
                    plot_xlstm_visualizations(
                        image_tensor=img_input.cpu(),
                        pseudo_attn_maps=pseudo_attn_maps,
                        surprise_map=surprise_map,
                        generated_tokens=tokens,
                        save_path_prefix=os.path.join(img_out_dir, "xlstm")
                    )
                except Exception as e:
                    import traceback
                    print(f"  [xLSTM Viz Failed] {img_name}: {e}")
                    traceback.print_exc()
                continue
            # ---------------------------------------------

            result = model(img_input, return_attention=True)

            # If model is not in attention mode, result is just logits
            if not isinstance(result, tuple):
                print(f"  [skip] {img_name} — model did not return attention maps.")
                continue

            logits, attn_maps = result                          # attn_maps: list[(1,L)]
            if not attn_maps:
                print(f"  [skip] {img_name} — empty attention map list.")
                continue

            pred_indices = logits.argmax(dim=1)[0].cpu()       # (T,)

            # Spatial grid dimensions
            L      = attn_maps[0].shape[1]
            if getattr(model, 'attn_type', None) == 'adaptive':
                L -= 1
            grid_h = int(math.sqrt(L))
            grid_w = int(math.ceil(L / grid_h))

            # Load original image
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                print(f"  [skip] Original image not found: {img_path}")
                continue

            # Decode tokens (stop at EOS)
            tokens = []
            valid_attn_maps = []
            words = []
            for t, idx in enumerate(pred_indices):
                idx_val = idx.item()
                if idx_val == getattr(tokenizer, 'eos_idx', -1):
                    break
                # Skip padding, sos, and unk tokens for better visualization as requested
                if idx_val in [getattr(tokenizer, 'pad_idx', -1), getattr(tokenizer, 'sos_idx', -1)]:
                    continue
                if hasattr(tokenizer, 'unk_idx') and idx_val == tokenizer.unk_idx:
                    continue
                    
                word = tokenizer.decode([idx_val]).strip() or f"tok{idx_val}"
                if not word or word == "<UNK>" or word == "[UNK]":
                    continue
                    
                tokens.append(idx_val)
                words.append(word)
                if t < len(attn_maps):
                    valid_attn_maps.append(attn_maps[t])

            if not tokens:
                print(f"  [skip] {img_name} — empty prediction after filtering.")
                continue

            n_words = min(len(tokens), len(valid_attn_maps))

            # ----------------------------------------------------------
            # Helper for Smooth Scaling Like "Show Attend and Tell"
            # ----------------------------------------------------------
            import scipy.ndimage as ndimage
            def get_smooth_alpha_map(alpha_raw, grid_h, grid_w, out_w, out_h):
                alpha_grid = alpha_raw.reshape(grid_h, grid_w)
                # Paper interpolation: Smooth the grid first
                alpha_grid = ndimage.gaussian_filter(alpha_grid, sigma=1.0)
                alpha_pil  = PILImage.fromarray((alpha_grid * 255).astype('uint8')).resize((out_w, out_h), resample=PILImage.BICUBIC)
                alpha_arr  = np.array(alpha_pil) / 255.0
                alpha_norm = (alpha_arr - alpha_arr.min()) / (alpha_arr.max() - alpha_arr.min() + 1e-8)
                # Additional blur for perfectly smooth glowing look
                alpha_norm = ndimage.gaussian_filter(alpha_norm, sigma=15.0)
                # re-normalize after heavy blur
                return (alpha_norm - alpha_norm.min()) / (alpha_norm.max() - alpha_norm.min() + 1e-8)

            # ----------------------------------------------------------
            # Per-word individual images (for soft / adaptive)
            # ----------------------------------------------------------
            img_stem    = os.path.splitext(img_name)[0]
            img_out_dir = os.path.join(attn_dir, f"{img_stem}_outputs")
            os.makedirs(img_out_dir, exist_ok=True)

            if model.attn_type in ['soft', 'adaptive']:
                for t in range(n_words):
                    word    = words[t]
                    alpha   = valid_attn_maps[t][0][:L].numpy()              # (L,)
                    
                    alpha_norm = get_smooth_alpha_map(alpha, grid_h, grid_w, orig_w, orig_h)
                    
                    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                    if model.attn_type == 'adaptive':
                        beta_t = valid_attn_maps[t][0][-1].item()
                        vis_prob = 1.0 - beta_t
                        # Fade the brightness based on how much it attends to the image
                        display_img = (orig_arr * (alpha_norm[..., np.newaxis] * vis_prob * 0.9 + 0.1)).astype(np.uint8)
                        ax.imshow(display_img)
                        ax.set_title(f'"{word}" (p_img={vis_prob:.2f})', fontsize=11, fontweight='bold')
                    else:
                        # True Show Attend and Tell style: grayscale image brightly illuminated where attended
                        gray_img = np.dot(orig_arr[..., :3], [0.2989, 0.5870, 0.1140])
                        gray_img3 = np.stack((gray_img,) * 3, axis=-1)
                        display_img = (gray_img3 * (alpha_norm[..., np.newaxis] * 0.85 + 0.15)).astype(np.uint8)
                        ax.imshow(display_img)
                        ax.set_title(f'"{word}"', fontsize=13, fontweight='bold')
                        
                    ax.axis('off')
                    plt.tight_layout(pad=0.3)
                    safe_word = "".join(c if c.isalnum() else "_" for c in word)[:20]
                    save_path = os.path.join(img_out_dir, f"word_{t:02d}_{safe_word}.png")
                    plt.savefig(save_path, dpi=90, bbox_inches='tight')
                    plt.close(fig)

            # ----------------------------------------------------------
            # Composite grids / multi-color visualization
            # ----------------------------------------------------------
            if model.attn_type == 'soft':
                n_cells  = n_words + 1
                n_cols   = min(5, n_cells)
                n_rows   = math.ceil(n_cells / n_cols)

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 3.0))
                axes = np.array(axes).reshape(-1)

                axes[0].imshow(orig_arr)
                axes[0].set_title("Input", fontsize=10, fontweight='bold')
                axes[0].axis('off')

                for t in range(n_words):
                    word    = words[t]
                    alpha   = valid_attn_maps[t][0][:L].numpy()
                    alpha_norm = get_smooth_alpha_map(alpha, grid_h, grid_w, orig_w, orig_h)
                    
                    ax = axes[t + 1]
                    gray_img = np.dot(orig_arr[..., :3], [0.2989, 0.5870, 0.1140])
                    gray_img3 = np.stack((gray_img,) * 3, axis=-1)
                    display_img = (gray_img3 * (alpha_norm[..., np.newaxis] * 0.85 + 0.15)).astype(np.uint8)
                    ax.imshow(display_img)
                    ax.set_title(f'"{word}"', fontsize=9)
                    ax.axis('off')

                for ax in axes[n_words + 1:]:
                    ax.axis('off')

                pred_display = pred_str[:80] + ("…" if len(pred_str) > 80 else "")
                plt.suptitle(f"Pred: {pred_display}", fontsize=8, y=1.01)
                plt.tight_layout(pad=0.5)

                grid_path = os.path.join(img_out_dir, "attention_grid.png")
                plt.savefig(grid_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: {img_out_dir}/")
                
            elif model.attn_type == 'adaptive':
                # --- 1) Grid of solid color overlays (like soft, instead of single overlap) ---
                import matplotlib.colors as mcolors
                
                # Colors more similar to the paper (Blue, Red, Green, Orange, Purple, Brown, Pink)
                paper_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
                colors_list = paper_colors * (n_words // len(paper_colors) + 1)
                
                n_cells  = n_words + 1
                n_cols   = min(5, n_cells)
                n_rows   = math.ceil(n_cells / n_cols)
                
                fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 3.0))
                axes2 = np.array(axes2).reshape(-1)
                
                axes2[0].imshow(orig_arr)
                axes2[0].set_title("Input", fontsize=10, fontweight='bold')
                axes2[0].axis('off')
                
                for t in range(n_words):
                    word = words[t]
                    alpha = valid_attn_maps[t][0][:L].numpy()
                    alpha_norm = get_smooth_alpha_map(alpha, grid_h, grid_w, orig_w, orig_h)
                    vis_prob = 1.0 - valid_attn_maps[t][0][-1].item()
                    
                    color_hex = colors_list[t]
                    c = mcolors.to_rgb(color_hex)
                    
                    overlay = orig_arr.copy() / 255.0
                    mask = (alpha_norm > np.percentile(alpha_norm, 80)) & (vis_prob > 0.35)
                    
                    for i_c in range(3):
                        # Tint the image with the specific cell's color
                        overlay[..., i_c] = np.where(mask, overlay[..., i_c]*0.4 + c[i_c]*0.6, overlay[..., i_c] * 0.9)
                        
                    ax = axes2[t + 1]
                    ax.imshow(np.clip(overlay, 0, 1))
                    ax.set_title(f'"{word}"', fontsize=12, color=color_hex, fontweight='bold')
                    ax.axis('off')
                
                for ax in axes2[n_words + 1:]:
                    ax.axis('off')
                    
                plt.tight_layout(pad=0.5)
                multicolor_grid_path = os.path.join(img_out_dir, "adaptive_colored_grid.png")
                plt.savefig(multicolor_grid_path, dpi=120, bbox_inches='tight')
                plt.close(fig2)

                # --- 2) Line graph of visual grounding probabilities with thumbnails ---
                fig3 = plt.figure(figsize=(max(10, n_words * 1.5), 5))
                import matplotlib.gridspec as gridspec
                gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2.5], height_ratios=[2, 1.2], figure=fig3)
                
                ax_orig = fig3.add_subplot(gs[:, 0])
                ax_orig.imshow(orig_arr)
                ax_orig.axis('off')
                
                ax_line = fig3.add_subplot(gs[0, 1])
                vis_probs = [(1.0 - valid_attn_maps[t][0][-1].item()) for t in range(n_words)]
                ax_line.plot(range(n_words), vis_probs, marker='o', linestyle='-', linewidth=2, markersize=8, color='#4A90E2')
                ax_line.set_ylim(-0.05, 1.05)
                ax_line.set_xlim(-0.5, n_words - 0.5)
                ax_line.set_xticks([])
                ax_line.grid(True, axis='y', linestyle='--', alpha=0.7)
                for i, p in enumerate(vis_probs):
                    ax_line.text(i, p + 0.05 if p < 0.8 else p - 0.15, f"{p:.3f}", ha='center', va='bottom', fontsize=10)
                
                gs_thumbs = gridspec.GridSpecFromSubplotSpec(1, n_words, subplot_spec=gs[1, 1], wspace=0.1)
                import matplotlib.cm as cm
                jet = cm.get_cmap('jet')
                for t in range(n_words):
                    ax_thumb = fig3.add_subplot(gs_thumbs[0, t])
                    alpha = valid_attn_maps[t][0][:L].numpy()
                    alpha_norm = get_smooth_alpha_map(alpha, grid_h, grid_w, orig_w, orig_h)
                    
                    colored_alpha = jet(alpha_norm)[..., :3]
                    # To mimic the paper thumbnail: original image overlaid with soft jet heatmap
                    blended = (orig_arr/255.0 * 0.5 + colored_alpha * 0.5)
                    blended = np.clip(blended, 0, 1)
                    
                    ax_thumb.imshow(blended)
                    ax_thumb.axis('off')
                    ax_thumb.set_title(words[t], y=-0.4, fontsize=12)
                
                plt.subplots_adjust(wspace=0.05, hspace=0) 
                sentinel_path = os.path.join(img_out_dir, "adaptive_sentinel_plot.png")
                plt.savefig(sentinel_path, dpi=120, bbox_inches='tight')
                plt.close(fig3)
                
                print(f"  Saved Adaptive plots: {img_out_dir}/")


# ---------------------------------------------------------------------------
#  Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, crit, dataloader, epoch, grad_clip=5.0):
    model.train()
    total_loss = 0
    grad_norms = []
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}", mininterval=30.0)

    for img, caption, _ in progress_bar:
        img, caption = img.to(DEVICE), caption.to(DEVICE)

        optimizer.zero_grad()
        pred = model(img, caption)
        loss = crit(pred, caption[:, 1:])

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

    sample_indices   = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    samples_to_print = []          # (pred_str, ref_strs, img_name, img_tensor_cpu)

    with torch.no_grad():
        for i, (img, caption, img_names) in enumerate(
            tqdm(dataloader, desc="Eval", mininterval=30.0)
        ):
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
            img_cpu      = img.cpu()                            # keep for attention viz

            for b in range(img.size(0)):
                pred_str      = convert_indices_to_string(pred_indices[b], tokenizer)
                actual_img_id = filename_to_img_id[img_names[b]]
                ref_strs      = dataloader.dataset.image_captions[actual_img_id]

                all_preds.append(pred_str)
                all_refs.append(ref_strs)
                all_refs_old.append([ref_strs[0]])

                global_idx = i * dataloader.batch_size + b
                if global_idx in sample_indices:
                    samples_to_print.append(
                        (pred_str, ref_strs, img_names[b], img_cpu[b])
                    )

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
        metrics = {k: 0 for k in [
            "BLEU-1", "BLEU-2", "ROUGE-L", "METEOR", "CIDEr",
            "BLEU-1_old", "BLEU-2_old", "ROUGE-L_old", "METEOR_old", "CIDEr_old"
        ]}

    print("\n--- Evaluation Samples ---")
    for s_pred, s_refs, s_img, _ in samples_to_print:
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


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    final_cfg = parse_args()
    cfg = final_cfg

    run_name = f"{cfg.encoder}_{cfg.decoder_type}_d{cfg.decoder_dim}_l{cfg.decoder_layers}_{cfg.text_level}"
    if cfg.attn_type:
        run_name += f"_attn_{cfg.attn_type}_{cfg.attn_dim}"
    if cfg.clip_embeddings:
        suffix    = "_frozen" if cfg.freeze_embeddings else "_unfrozen"
        run_name += f"_clip_embed{suffix}"

    run = wandb.init(
        project=cfg.project,
        config=vars(cfg),
        name=run_name,
    )
    cfg = wandb.config

    print(
        f"Config: encoder={cfg.encoder}, text_level={cfg.text_level}, mode={cfg.mode}, "
        f"lr={cfg.lr}, batch_size={cfg.batch_size}, epochs={cfg.epochs}, "
        f"attn_type={cfg.attn_type}, attn_dim={cfg.attn_dim}"
    )

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

    # --- Tokenizer ---
    print(f"Initializing {cfg.text_level} tokenizer...")
    if cfg.text_level == 'char':
        tokenizer = CharTokenizer()
    elif cfg.text_level == 'subword':
        tokenizer = SubwordTokenizer()
    elif cfg.text_level == 'word':
        vocab = build_word_vocab(TRAIN_ANN)
        tokenizer = WordTokenizer(vocab=vocab)
    else:
        raise ValueError(f"Unknown text_level: {cfg.text_level}")

    # --- Datasets ---
    print("Loading datasets...")
    encoder_source = ENCODER_CONFIGS.get(cfg.encoder, ('hf',))[0]
    if encoder_source == 'clip':
        img_mean = VizWizDataset.CLIP_MEAN
        img_std  = VizWizDataset.CLIP_STD
    else:
        img_mean = VizWizDataset.IMAGENET_MEAN
        img_std  = VizWizDataset.IMAGENET_STD

    if cfg.mode == "search":
        dataset_train = VizWizDataset(
            TRAIN_ANN, TRAIN_IMG_DIR, split="train_search", mode=cfg.mode,
            img_mean=img_mean, img_std=img_std, tokenizer=tokenizer
        )
        dataset_valid = VizWizDataset(
            TRAIN_ANN, TRAIN_IMG_DIR, split="val_search", mode=cfg.mode,
            img_mean=img_mean, img_std=img_std, tokenizer=tokenizer
        )
        val_img_dir = TRAIN_IMG_DIR
    else:
        dataset_train = VizWizDataset(
            TRAIN_ANN, TRAIN_IMG_DIR, split="train", mode=cfg.mode,
            img_mean=img_mean, img_std=img_std, tokenizer=tokenizer
        )
        dataset_valid = VizWizDataset(
            VAL_ANN, VAL_IMG_DIR, split="val", mode=cfg.mode,
            img_mean=img_mean, img_std=img_std, tokenizer=tokenizer
        )
        val_img_dir = VAL_IMG_DIR

    dataloader_train = DataLoader(
        dataset_train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True
    )
    dataloader_valid = DataLoader(
        dataset_valid, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers
    )

    # --- Model ---
    print(f"Initializing model: encoder={cfg.encoder}, decoder={cfg.decoder_type}, "
          f"attention={cfg.attn_type} ...")
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
        freeze_embeddings=cfg.freeze_embeddings,
        attn_type=cfg.attn_type,
        attn_dim=cfg.attn_dim,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update(
        {"trainable_params": trainable_params, "vocab_size": tokenizer.vocab_size},
        allow_val_change=True
    )

    scheduler = None
    if cfg.scheduler == "plateau":
        print("Using ReduceLROnPlateau scheduler")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    primary_metric_name = "METEOR"
    best_primary = -1.0
    best_epoch   = -1

    for epoch in range(1, cfg.epochs + 1):
        loss, avg_grad_norm, max_grad_norm = train_one_epoch(
            model, optimizer, crit, dataloader_train, epoch, grad_clip=cfg.grad_clip
        )
        print(
            f"End of Epoch {epoch} - Train Loss: {loss:.4f} | "
            f"Grad Norm (avg/max): {avg_grad_norm:.3f} / {max_grad_norm:.3f}"
        )

        metrics, eval_samples = eval_epoch(model, dataloader_valid, crit, tokenizer)

        # Save visual samples (done once at epoch 1)
        epoch_samples = []
        for pred, refs, img_name, _ in eval_samples:
            epoch_samples.append({"image_name": img_name, "references": refs, "prediction": pred})
            if epoch == 1:
                src_img = os.path.join(val_img_dir, img_name)
                if os.path.exists(src_img):
                    shutil.copy(src_img, os.path.join(samples_dir, img_name))

        caption_history[epoch] = epoch_samples
        with open(history_file, "w") as f:
            json.dump(caption_history, f, indent=4)

        print("Validation Metrics:")
        print(
            f"  [New] BLEU-1: {metrics.get('BLEU-1', 0):.2f}% | "
            f"BLEU-2: {metrics.get('BLEU-2', 0):.2f}% | "
            f"ROUGE-L: {metrics.get('ROUGE-L', 0):.2f}% | "
            f"METEOR: {metrics.get('METEOR', 0):.4f}% | "
            f"CIDEr: {metrics.get('CIDEr', 0):.2f}"
        )
        print(
            f"  [Old] BLEU-1: {metrics.get('BLEU-1_old', 0):.2f}% | "
            f"BLEU-2: {metrics.get('BLEU-2_old', 0):.2f}% | "
            f"ROUGE-L: {metrics.get('ROUGE-L_old', 0):.2f}% | "
            f"METEOR: {metrics.get('METEOR_old', 0):.4f}% | "
            f"CIDEr: {metrics.get('CIDEr_old', 0):.2f}"
        )
        print(
            f"  Val Loss: {metrics.get('val_loss', 0):.4f} | "
            f"FPS: {metrics.get('compute/fps', 0):.1f} | "
            f"Avg latency: {metrics.get('compute/avg_batch_latency_ms', 0):.1f}ms | "
            f"Max VRAM: {metrics.get('compute/max_vram_gb', 0):.3f}GB"
        )

        wandb.log({
            "epoch":         epoch,
            "train_loss":    loss,
            "lr":            optimizer.param_groups[0]['lr'],
            "grad_norm/avg": avg_grad_norm,
            "grad_norm/max": max_grad_norm,
            "best_METEOR":   best_primary,
            "best_epoch":    best_epoch,
            **metrics,
        })

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics.get(primary_metric_name, 0))
            else:
                scheduler.step()

        if metrics.get(primary_metric_name, 0) > best_primary:
            best_primary = metrics[primary_metric_name]
            best_epoch   = epoch

            torch.save(model.state_dict(), ckpt_path)

            best_metrics_payload = {
                **metrics,
                "best_epoch": best_epoch,
            }
            with open(os.path.join(run_output_dir, "best_metrics.json"), "w") as f:
                json.dump(best_metrics_payload, f, indent=4)

            print(
                f"  -> New best {primary_metric_name}: {best_primary:.2f} "
                f"(epoch {best_epoch}) — checkpoint saved."
            )

            wandb.run.summary[f"best_{primary_metric_name}"] = best_primary
            wandb.run.summary["best_epoch"] = best_epoch
            for k, v in metrics.items():
                if k.startswith("compute/"):
                    wandb.run.summary[k] = v

        # --- Attention visualizations for all epochs ---
        is_new_best = (epoch == best_epoch)
        if cfg.attn_type:
            generate_attention_visualizations(
                model=model,
                tokenizer=tokenizer,
                sample_data=eval_samples,   # list of (pred, refs, img_name, img_tensor)
                epoch=epoch,
                run_output_dir=run_output_dir,
                img_dir=val_img_dir,
            )

    wandb.finish()


if __name__ == "__main__":
    main()
