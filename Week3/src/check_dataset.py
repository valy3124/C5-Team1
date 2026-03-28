import os
import json
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Week 2 plotting style constants
TITLE_FS    = 24
LABEL_FS    = 18
TICK_FS     = 16
LEGEND_FS   = 16
BAR_FS      = 15

matplotlib.rcParams.update({
    "font.size":        LABEL_FS,
    "axes.titlesize":   TITLE_FS,
    "axes.labelsize":   LABEL_FS,
    "xtick.labelsize":  TICK_FS,
    "ytick.labelsize":  TICK_FS,
    "legend.fontsize":  LEGEND_FS,
})

# Color palette from Week 2
COLORS = {
    "pretrained": "#8ecae6",
    "finetuned":  "#023047",
    "bbox":       "#1f4e79",
    "point":      "#2d9e6b",
    "text":       "#2a2725",
    "mix":        "#7b2d8b"
}

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'VizWiz')
img_train_path = os.path.join(base_path, 'images', 'train')
img_val_path = os.path.join(base_path, 'images', 'val')
img_test_path = os.path.join(base_path, 'images', 'test')

ann_train_path = os.path.join(base_path, 'annotations', 'train.json')
ann_val_path = os.path.join(base_path, 'annotations', 'val.json')

print("--- Image Counts ---")
try:
    print(f"Train images: {len(os.listdir(img_train_path))}")
except Exception as e:
    print(f"Train images error: {e}")

try:
    print(f"Validation images: {len(os.listdir(img_val_path))}")
except Exception as e:
    print(f"Validation images error: {e}")

try:
    print(f"Test images: {len(os.listdir(img_test_path))}")
except Exception as e:
    print(f"Test images error: {e}")

print("\n--- Annotation Counts ---")
train_ann = None
val_ann = None

try:
    with open(ann_train_path, 'r') as f:
        train_ann = json.load(f)
        print(f"Train Annotations Info:")
        print(f"  Images in JSON: {len(train_ann.get('images', []))}")
        print(f"  Captions in JSON: {len(train_ann.get('annotations', []))}")
except Exception as e:
    print(f"Train annotations error: {e}")

try:
    with open(ann_val_path, 'r') as f:
        val_ann = json.load(f)
        print(f"Validation Annotations Info:")
        print(f"  Images in JSON: {len(val_ann.get('images', []))}")
        print(f"  Captions in JSON: {len(val_ann.get('annotations', []))}")
except Exception as e:
    print(f"Validation annotations error: {e}")

# ============ PLOTTING SECTION ============
if train_ann and val_ann:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ---- Data Prep: Histogram of captions per image ----
    caption_counts_train = {img['id']: 0 for img in train_ann['images']}
    for ann in train_ann['annotations']:
        if not ann.get('is_precanned', False) and not ann.get('is_rejected', False):
            caption_counts_train[ann['image_id']] += 1
            
    caption_counts_val = {img['id']: 0 for img in val_ann['images']}
    for ann in val_ann['annotations']:
        if not ann.get('is_precanned', False) and not ann.get('is_rejected', False):
            caption_counts_val[ann['image_id']] += 1
            
    all_counts = list(caption_counts_train.values()) + list(caption_counts_val.values())
    caption_dist = Counter(all_counts)

    train_dist = Counter(caption_counts_train.values())
    val_dist = Counter(caption_counts_val.values())
    
    print("\n--- Caption Count Distribution ---")
    for cap_count in sorted(caption_dist.keys()):
        print(f"  {cap_count} captions: {caption_dist[cap_count]} images (Train: {train_dist[cap_count]}, Val: {val_dist[cap_count]})")
        
    # Plot 1: Captions per Image (Stacked)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Distribution: Captions per Image(Full Data Split)", fontsize=TITLE_FS, fontweight="bold", pad=14)
    caption_nums = sorted(caption_dist.keys())
    
    tr_counts = [train_dist[num] for num in caption_nums]
    va_counts = [val_dist[num] for num in caption_nums]
    
    width = 0.6
    bars_tr = ax.bar([str(num) for num in caption_nums], tr_counts, width, label='Training', color=COLORS["pretrained"], edgecolor="white")
    bars_va = ax.bar([str(num) for num in caption_nums], va_counts, width, bottom=tr_counts, label='Validation', color=COLORS["point"], edgecolor="white")
    
    ax.set_xlabel("Number of Valid Captions per Image", fontsize=LABEL_FS)
    ax.set_ylabel("Number of Images", fontsize=LABEL_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=LEGEND_FS, frameon=False)
    
    max_h = max(caption_dist[num] for num in caption_nums)
    for i, (tr, va) in enumerate(zip(tr_counts, va_counts)):
        total = tr + va
        if total > 0:
            # Total value on top
            ax.text(i, total + max_h*0.015, str(total), 
                    ha="center", va="bottom", fontsize=BAR_FS, fontweight="bold", color="black")
            
            # Print value inside if it is large enough
            if tr > max_h * 0.04:
                ax.text(i, tr/2, str(tr), ha='center', va='center', fontweight='bold', fontsize=BAR_FS-2, color='black')
            if va > max_h * 0.04:
                ax.text(i, tr + va/2, str(va), ha='center', va='center', fontweight='bold', fontsize=BAR_FS-2, color='black')
                
    plt.tight_layout()
    p1 = os.path.join(out_dir, "dataset_captions_dist.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {p1}")

    # ---- Plot 2: Dataset distribution (Search vs Full) ----
    valid_img_ids_train = sorted([img_id for img_id, count in caption_counts_train.items() if count > 0])
    valid_img_ids_val = sorted([img_id for img_id, count in caption_counts_val.items() if count > 0])
    test_images = len(os.listdir(img_test_path)) if os.path.exists(img_test_path) else 0
    
    random.seed(42)
    valid_ids_search = valid_img_ids_train.copy()
    random.shuffle(valid_ids_search)
    num_train_search = int(len(valid_ids_search) * 0.9)
    train_search = num_train_search
    val_search = len(valid_ids_search) - num_train_search
    
    train_full = len(valid_img_ids_train)
    val_full = len(valid_img_ids_val)
    
    print(f"\n--- Dataset Distribution ---")
    print(f"Search Mode: Train={train_search}, Val={val_search}")
    print(f"Full Mode: Train={train_full}, Val={val_full}, Test={test_images}")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Dataset Distribution: Search vs. Full Mode", fontsize=TITLE_FS, fontweight="bold", pad=14)
    
    modes = ['Search Mode', 'Full Mode']
    train_counts = [train_search, train_full]
    val_counts = [val_search, val_full]
    test_counts = [0, test_images]
    
    x = np.arange(len(modes))
    width = 0.5
    
    # Stacked bars
    bars_tr = ax.bar(x, train_counts, width, label='Training', color=COLORS["pretrained"], edgecolor="white")
    bars_va = ax.bar(x, val_counts, width, bottom=train_counts, label='Validation', color=COLORS["point"], edgecolor="white")
    #bars_te = ax.bar(x, test_counts, width, bottom=np.array(train_counts) + np.array(val_counts), label='Testing', color=COLORS["text"], edgecolor="white")
    
    ax.set_ylabel("Number of Images", fontsize=LABEL_FS)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=TICK_FS + 2)
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc='upper left')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Add value labels inside the stacked bars
    for i, (tr, va, te) in enumerate(zip(train_counts, val_counts, test_counts)):
        ax.text(i, tr/2, str(int(tr)), ha='center', va='center', fontweight='bold', fontsize=BAR_FS, color='black')
        ax.text(i, tr + va/2, str(int(va)), ha='center', va='center', fontweight='bold', fontsize=BAR_FS, color='black')
        if te > 0:
            ax.text(i, tr + va + te/2, str(int(te)), ha='center', va='center', fontweight='bold', fontsize=BAR_FS, color='black')
                        
    plt.tight_layout()
    p3 = os.path.join(out_dir, "dataset_splits_dist.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {p3}")
