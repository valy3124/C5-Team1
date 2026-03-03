"""
qualitative_deart.py — Qualitative evaluation for DEArt domain shift.

Shows side-by-side:
  Row 0: Ground Truth (DEArt annotations, person class)
  Row 1: KITTI-MOTS model (zero-shot on DEArt)
  Row 2: Fine-tuned DEArt model (LoRA adapted)

Usage (from Week1/):
    python src/fine_tune/plots_for_slides/qualitative_deart.py
"""

import os
import sys
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[3]   # Week1/
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

import datasets as project_datasets   # Week1/src/datasets.py

KITTI_WEIGHTS  = str(BASE_DIR / "results/detr/detr_Freeze_L3_LR_0.0001_x9kpvkwf/best_model.pth")
DEART_LORA_DIR = str(BASE_DIR / "results/detr/detr_base_Frz3_LR5.0e-05_WD1.0e-04_Sch_cosine_GC_0.8_9gptqe2i")
DEART_ROOT     = str(BASE_DIR / "src/DEArt/")
RESULTS_DIR    = str(BASE_DIR / "results/qualitative_deart")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
NUM_IMAGES     = 20        # how many images to visualise
CONF_THRESHOLD = 0.3       # detection confidence threshold
MAX_SIZE       = 800       # match training resize
SEED           = 42
SPLIT_RATIO    = 0.8       # same as training

# Label mapping: model outputs DETR label 2 (pedestrian slot) → display as "Person"
PERSON_DETR_LABEL = 2      # the "pedestrian" slot in the KITTI-trained model
# For GT: DEArt raw class_id=1 (person)
GT_PERSON_ID = 1

# ─── Colours ──────────────────────────────────────────────────────────────────
GT_COLOR       = "#2ecc71"   # green
KITTI_COLOR    = "#e74c3c"   # red
DEART_COLOR    = "#3498db"   # blue


# ─── Model helpers ────────────────────────────────────────────────────────────

def load_kitti_model(device):
    """Load DETR fine-tuned on KITTI-MOTS (3 classes: background=0, car=1, pedestrian=2).
    HuggingFace DETR adds 1 extra no-object output, so num_labels=3 → classifier [4, 256].
    """
    id2label = {0: "background", 1: "car", 2: "pedestrian"}
    label2id = {v: k for k, v in id2label.items()}

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=3,
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id,
    )
    state_dict = torch.load(KITTI_WEIGHTS, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[KITTI model] missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device).eval()
    return model


def load_deart_model(device):
    """Load DEArt LoRA fine-tuned model."""
    from peft import PeftModel

    id2label = {0: "background", 1: "car", 2: "pedestrian"}
    label2id = {v: k for k, v in id2label.items()}

    # 1. Build base DETR architecture (same 3-class setup as KITTI training)
    base_model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=3,
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id,
    )
    # 2. Warm-start with KITTI weights (same starting point used during DEArt fine-tuning)
    state_dict = torch.load(KITTI_WEIGHTS, map_location="cpu")
    base_model.load_state_dict(state_dict, strict=False)

    # 3. Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, DEART_LORA_DIR)
    model = model.merge_and_unload()   # fuse adapters into weights → standard model
    print("[DEArt LoRA model] adapters merged and loaded.")
    model.to(device).eval()
    return model


def build_processor():
    return DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": MAX_SIZE, "longest_edge": MAX_SIZE},
    )


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, processor, img_np, device):
    """Return list of (box_xyxy, score) for the pedestrian/person class."""
    inputs = processor(images=[img_np], return_tensors="pt", do_rescale=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    h, w = img_np.shape[:2]
    target_sizes = torch.tensor([[h, w]], device=device)

    outputs = model(**inputs)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=CONF_THRESHOLD
    )[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if int(label.item()) == PERSON_DETR_LABEL:
            detections.append((box.cpu().tolist(), float(score.item())))
    return detections


# ─── Drawing ──────────────────────────────────────────────────────────────────

def draw_boxes(ax, img_np, detections, color, title, show_score=True):
    """detections: list of (box_xyxy, score) or (box_xyxy,) for GT."""
    ax.imshow(img_np)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6, color="white",
                 bbox=dict(facecolor="black", alpha=0.6, pad=4, boxstyle="round"))
    ax.axis("off")

    for item in detections:
        if len(item) == 2:
            box, score = item
        else:
            box, score = item, None

        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        label_txt = "Person"
        if show_score and score is not None:
            label_txt += f" {score:.2f}"

        ax.text(
            x1, max(y1 - 5, 0), label_txt,
            color="white", fontsize=8, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.7, pad=2, edgecolor="none")
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load models & processor ──────────────────────────────────────────────
    print("Loading KITTI model (zero-shot)...")
    kitti_model = load_kitti_model(device)

    print("Loading DEArt fine-tuned model...")
    deart_model = load_deart_model(device)

    processor = build_processor()

    # ── Load DEArt validation split ──────────────────────────────────────────
    print("Loading DEArt validation dataset...")
    # "dev" matches the split used during fine-tuning (mode=search → train/dev)
    val_ds = project_datasets.DEART(
        root=DEART_ROOT,
        split="dev",
        ann_source="xml",
        seed=SEED,
        split_ratio=SPLIT_RATIO,
    )
    print(f"Validation set size: {len(val_ds)} images")

    # ── Sample indices (evenly spread) ──────────────────────────────────────
    rng = random.Random(SEED)
    indices = list(range(len(val_ds)))
    if len(indices) > NUM_IMAGES:
        step = len(indices) // NUM_IMAGES
        indices = [indices[i * step] for i in range(NUM_IMAGES)]

    # ── Generate plots ───────────────────────────────────────────────────────
    for rank, idx in enumerate(indices):
        img_pil, raw_anns, meta = val_ds[idx]
        img_np = np.array(img_pil)

        # GT boxes (raw DEArt class_id = 1 for person, already correct)
        gt_det = []
        for ann in raw_anns:
            if int(getattr(ann, "class_id", -1)) == GT_PERSON_ID:
                box = getattr(ann, "bbox_xyxy", None)
                if box is not None:
                    gt_det.append((list(map(float, box)), None))  # (box, score=None)

        # Model inference (images already in 0-255 uint8, do_rescale=False requires float [0,1])
        img_float = img_np.astype(np.float32) / 255.0

        kitti_det = run_inference(kitti_model, processor, img_float, device)
        deart_det = run_inference(deart_model, processor, img_float, device)

        # ── Plot ─────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("#1a1a2e")

        draw_boxes(axes[0], img_np, gt_det,    GT_COLOR,    f"Ground Truth  ({len(gt_det)} persons)", show_score=False)
        draw_boxes(axes[1], img_np, kitti_det, KITTI_COLOR, f"KITTI zero-shot  ({len(kitti_det)} det)", show_score=True)
        draw_boxes(axes[2], img_np, deart_det, DEART_COLOR, f"DEArt fine-tuned  ({len(deart_det)} det)", show_score=True)

        plt.tight_layout(pad=1.5)

        seq   = meta.get("seq", "unk") if isinstance(meta, dict) else "unk"
        frame = meta.get("frame", idx) if isinstance(meta, dict) else idx
        img_path = meta.get("image_path", "")
        stem = Path(img_path).stem if img_path else f"img_{idx}"
        fname = f"{rank:03d}_{stem}.jpg"
        out_path = os.path.join(RESULTS_DIR, fname)
        plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  [{rank+1}/{len(indices)}] Saved {fname}  |  GT={len(gt_det)}  KITTI={len(kitti_det)}  DEArt={len(deart_det)}")

    print(f"\nDone. {len(indices)} images saved to:\n  {RESULTS_DIR}")


if __name__ == "__main__":
    main()
