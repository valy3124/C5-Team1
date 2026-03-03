"""
deart_stats.py — Generate dataset statistics plots for DEArt.
Run from Week1/ directory:
    python src/deart_stats.py
"""
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
ROOT     = Path("DEArt")
ANN_ROOT = ROOT / "annotations" / "annots_pub"
IMG_ROOT = ROOT / "images"
OUT_DIR  = Path("results/deart_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_MAPPING = {
    "angel": 1, "centaur": 1, "crucifixion": 1, "devil": 1,
    "god the father": 1, "judith": 1, "knight": 1, "monk": 1,
    "nude": 1, "person": 1, "shepherd": 1,
}
SEED       = 42
SPLIT_RATIO = 0.8

# ---------------------------------------------------------------------------
# 1. Collect stats
# ---------------------------------------------------------------------------
xml_stems  = {p.stem for p in ANN_ROOT.glob("*.xml")}
valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
img_dict   = {p.stem: p for p in IMG_ROOT.iterdir() if p.suffix in valid_exts}
valid_ids  = sorted(xml_stems.intersection(img_dict.keys()))

all_classes   = Counter()
person_classes = Counter()
widths, heights, ann_counts = [], [], []

# Split first so we can tag each image
rng      = random.Random(SEED)
all_ids  = list(valid_ids)
rng.shuffle(all_ids)
n_train  = int(len(all_ids) * SPLIT_RATIO)
n_val    = len(all_ids) - n_train
train_ids = set(all_ids[:n_train])
val_ids   = set(all_ids[n_train:])

person_classes_train = Counter()
person_classes_val   = Counter()

for vid in valid_ids:
    tree = ET.parse(ANN_ROOT / f"{vid}.xml")
    r    = tree.getroot()
    size = r.find("size")
    if size is not None:
        widths.append(int(size.find("width").text))
        heights.append(int(size.find("height").text))
    n_anns = 0
    for obj in r.findall("object"):
        name = obj.find("name").text
        all_classes[name] += 1
        if name in LABELS_MAPPING:
            person_classes[name] += 1
            n_anns += 1
            if vid in train_ids:
                person_classes_train[name] += 1
            else:
                person_classes_val[name] += 1
    ann_counts.append(n_anns)

# ---------------------------------------------------------------------------
# 2. Plot 1 — Person-class breakdown (bar chart)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
labels       = [k for k, _ in person_classes.most_common()]
counts_train = [person_classes_train[k] for k in labels]
counts_val   = [person_classes_val[k]   for k in labels]
x = np.arange(len(labels))
bars_train = ax.bar(x, counts_train, color="#08519c", edgecolor="white", label=f"Train ({n_train:,} images)")
bars_val   = ax.bar(x, counts_val,   color="#fd8d3c", edgecolor="white", label=f"Dev/Val ({n_val:,} images)", bottom=counts_train)
ax.set_yscale("log")
max_total = max(t + v for t, v in zip(counts_train, counts_val))
ax.set_ylim(top=max_total * 3)
# Total label on top (log-space offset)
for i, (t, v) in enumerate(zip(counts_train, counts_val)):
    total = t + v
    ax.text(i, total * 1.15, str(total),
            ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
ax.set_title("DEArt — Person Class by Split", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of annotations (log scale)")
ax.set_xlabel("Original class label")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_person_class_breakdown.png", dpi=150)
plt.close()
print("Saved 01_person_class_breakdown.png")

# ---------------------------------------------------------------------------
# 3. Plot 2 — Top-20 ALL classes (person vs ignored)
# ---------------------------------------------------------------------------
top20    = all_classes.most_common(20)
t_labels = [k for k, _ in top20]
t_counts = [v for _, v in top20]
t_colors = ["#2196F3" if k in LABELS_MAPPING else "#BDBDBD" for k in t_labels]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(t_labels, t_counts, color=t_colors, edgecolor="white")
ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
ax.set_title("DEArt — Top-20 Classes (blue = mapped to person, grey = ignored)", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of annotations")
plt.xticks(rotation=40, ha="right", fontsize=9)
blue_patch  = mpatches.Patch(color="#2196F3", label="Mapped → person")
grey_patch  = mpatches.Patch(color="#BDBDBD", label="Ignored")
ax.legend(handles=[blue_patch, grey_patch])
plt.tight_layout()
plt.savefig(OUT_DIR / "02_top20_all_classes.png", dpi=150)
plt.close()
print("Saved 02_top20_all_classes.png")

# ---------------------------------------------------------------------------
# 4. Plot 3 — Train / Val split (pie)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    [n_train, n_val],
    labels=[f"Train\n{n_train:,} images", f"Dev/Val\n{n_val:,} images"],
    autopct="%1.1f%%",
    colors=["#42A5F5", "#EF5350"],
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 12},
)
ax.set_title(f"DEArt — Train/Val Split\n(seed={SEED}, ratio={SPLIT_RATIO})", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_train_val_split.png", dpi=150)
plt.close()
print("Saved 03_train_val_split.png")

# ---------------------------------------------------------------------------
# 5. Plot 4 — Image size distribution (scatter width vs height)
# ---------------------------------------------------------------------------
if widths:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(widths, heights, alpha=0.05, s=4, color="#1565C0")
    ax.axvline(np.median(widths),  color="red",    linestyle="--", linewidth=1.2, label=f"Median W={int(np.median(widths))}px")
    ax.axhline(np.median(heights), color="orange", linestyle="--", linewidth=1.2, label=f"Median H={int(np.median(heights))}px")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("DEArt — Image Size Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_image_sizes.png", dpi=150)
    plt.close()
    print("Saved 04_image_sizes.png")

# ---------------------------------------------------------------------------
# 6. Plot 5 — Annotations per image histogram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(ann_counts, bins=40, color="#66BB6A", edgecolor="white")
ax.axvline(np.mean(ann_counts),   color="red",    linestyle="--", linewidth=1.5, label=f"Mean = {np.mean(ann_counts):.1f}")
ax.axvline(np.median(ann_counts), color="orange", linestyle="--", linewidth=1.5, label=f"Median = {np.median(ann_counts):.1f}")
ax.set_xlabel("Person annotations per image")
ax.set_ylabel("Number of images")
ax.set_title("DEArt — Distribution of Person Annotations per Image", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "05_annotations_per_image.png", dpi=150)
plt.close()
print("Saved 05_annotations_per_image.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n=== DEArt Summary ===")
print(f"Total images:            {len(valid_ids):,}")
print(f"Total person annotations:{sum(person_classes.values()):,}")
print(f"Total all-class anns:    {sum(all_classes.values()):,}")
print(f"Person classes:          {len(person_classes)}")
print(f"Ignored classes:         {len(all_classes) - len(person_classes)}")
print(f"Train images:            {n_train:,}")
print(f"Val images:              {n_val:,}")
if widths:
    print(f"Median image size:       {int(np.median(widths))} x {int(np.median(heights))} px")
    print(f"Max image size:          {max(widths)} x {max(heights)} px")
print(f"\nPlots saved to: {OUT_DIR.resolve()}")
