"""
Week2 SAM Fine-tuning Plot Utilities
=====================================
Generates comparison plots between pretrained and finetuned SAM models
across all prompt types: bbox, point, text, and mix.

Usage:
    python plot_utils.py \
        --finetuned_dir /path/to/final_finetuned \
        --eval_dir /path/to/results_eval \
        --output_dir /path/to/output_plots

Directory structure expected:
    final_finetuned/
        sam_bbox_*/  -> best_metrics.json (simple keys), config.yaml
        sam_point_*/ -> best_metrics.json (simple keys), config.yaml
        sam_text_*/  -> best_metrics.json (simple keys), config.yaml
        sam_mix_*/   -> best_metrics.json (prefixed keys: bbox_/point_/text_), config.yaml

    results_eval/
        eval_sam_metrics_validation.json         -> bbox pretrained
        eval_sam_metrics_point_validation.json   -> point pretrained
        eval_sam_metrics_text_validation.json    -> text pretrained
        eval_sam_metrics_mix_validation.json     -> mix pretrained (prefixed keys)
"""

import os
import json
import yaml
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path


# ---------------------------------------------------------------------------
# Global font sizes
# ---------------------------------------------------------------------------
TITLE_FS    = 24
SUPTITLE_FS = 24
LABEL_FS    = 18
TICK_FS     = 16
ANNOT_FS    = 22   # bigger heatmap cell numbers
HEAT_FS     = 22   # bigger heatmap axis ticks
LEGEND_FS   = 16
BAR_FS      = 15   # bigger bar value labels
DELTA_FS    = 14

matplotlib.rcParams.update({
    "font.size":        LABEL_FS,
    "axes.titlesize":   TITLE_FS,
    "axes.labelsize":   LABEL_FS,
    "xtick.labelsize":  TICK_FS,
    "ytick.labelsize":  TICK_FS,
    "legend.fontsize":  LEGEND_FS,
})


# ---------------------------------------------------------------------------
# Shared color palette
# ---------------------------------------------------------------------------
COLORS = {
    "pretrained": "#8ecae6",
    "finetuned":  "#023047",
    "bbox":       "#1f4e79",
    "point":      "#2d9e6b",
    "text":       "#e86c1f",
    "mix":        "#7b2d8b",
    "car":        "#1f4e79",
    "person":     "#9fd3c7",
}

PROMPT_TYPES  = ["bbox", "point", "text", "mix"]
PROMPT_LABELS = {"bbox": "Bbox", "point": "Point", "text": "Text", "mix": "Mix"}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _detect_prompt_type(config: dict, dir_name: str) -> str:
    pt = config.get("training", {}).get("prompt_type", None)
    if pt:
        return pt
    for p in PROMPT_TYPES:
        if p in dir_name.lower():
            return p
    return "bbox"


def load_finetuned_models(finetuned_dir: str) -> dict:
    models = {}
    base = Path(finetuned_dir)
    if not base.exists():
        print(f"[WARNING] Finetuned directory not found: {finetuned_dir}")
        return models

    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_file = run_dir / "best_metrics.json"
        config_file  = run_dir / "config.yaml"
        if not metrics_file.exists() or not config_file.exists():
            print(f"[SKIP] {run_dir.name}: missing best_metrics.json or config.yaml")
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)
        with open(config_file) as f:
            config = yaml.safe_load(f)

        pt    = _detect_prompt_type(config, run_dir.name)
        label = PROMPT_LABELS.get(pt, pt.capitalize())
        models[pt] = {"metrics": metrics, "config": config, "label": label, "prompt_type": pt}

    return models


def load_pretrained_evals(eval_dir: str) -> dict:
    eval_map = {
        "bbox":  "eval_sam_metrics_validation.json",
        "point": "eval_sam_metrics_point_validation.json",
        "text":  "eval_sam_metrics_text_validation.json",
        "mix":   "eval_sam_metrics_mix_validation.json",
    }
    evals = {}
    base = Path(eval_dir)
    for pt, filename in eval_map.items():
        path = base / filename
        if path.exists():
            with open(path) as f:
                evals[pt] = json.load(f)
        else:
            print(f"[WARNING] Pretrained eval not found: {path}")
    return evals


def _get(metrics: dict, key: str, default: float = 0.0) -> float:
    return metrics.get(key, default)


def _get_mix(metrics: dict, sub_type: str, key: str, default: float = 0.0) -> float:
    return metrics.get(f"{sub_type}_{key}", default)


def _mix_pre_avg(pre_metrics: dict, key: str) -> float:
    """Average a metric across bbox/point/text sub-types from the mix pretrained eval."""
    vals = [_get_mix(pre_metrics, st, key) for st in ["bbox", "point", "text"]]
    return float(np.mean(vals))


def _mix_ft_avg(ft_metrics: dict, key: str) -> float:
    """Average a metric across bbox/point/text sub-types from the mix finetuned metrics."""
    vals = [_get_mix(ft_metrics, st, key) for st in ["bbox", "point", "text"]]
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Delta annotation on top of a bar
# ---------------------------------------------------------------------------

def _annotate_delta(ax, bar, delta: float, bar_fs: int = DELTA_FS):
    """Draw a rounded box above a bar showing +/- delta.
    - green for positive, red for negative, orange for zero.
    - positioned well above the bar value label to avoid overlap.
    """
    h = bar.get_height()
    x = bar.get_x() + bar.get_width() / 2
    sign = "+" if delta >= 0 else ""
    if abs(delta) < 0.05:          # effectively 0.0
        color = "#e86c1f"          # orange
    elif delta > 0:
        color = "#2d9e6b"          # green
    else:
        color = "#c0392b"          # red
    text = f"{sign}{delta:.1f}"
    ax.text(
        x, h + 7.0, text,          # shifted higher to avoid overlap
        ha="center", va="bottom",
        fontsize=bar_fs, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none", alpha=0.93),
    )


# ---------------------------------------------------------------------------
# Plot 1 – mAP comparison: all 4 prompt types, pretrained vs finetuned,
#          with delta boxes on finetuned bars
# ---------------------------------------------------------------------------

def plot_input_types_comparison(
    finetuned_models: dict,
    pretrained_evals: dict,
    output_dir: str,
):
    """
    One grouped bar per prompt type (bbox / point / text / mix).
    Each group has 2 bars: pretrained mAP and finetuned mAP.
    A delta box floats above each finetuned bar.
    """
    key = "overall/AP_segm"
    types = ["bbox", "point", "text", "mix"]

    pre_vals, ft_vals = [], []
    for pt in types:
        if pt == "mix":
            pre = _get(pretrained_evals.get("mix", {}), "avg_overall/AP_segm",
                       _mix_pre_avg(pretrained_evals.get("mix", {}), "overall/AP_segm"))
            ft  = _get(finetuned_models.get("mix", {}).get("metrics", {}), "overall/AP_segm")
        else:
            pre = _get(pretrained_evals.get(pt, {}), key)
            ft  = _get(finetuned_models.get(pt, {}).get("metrics", {}), key)
        pre_vals.append(pre * 100)
        ft_vals.append(ft * 100)

    x     = np.arange(len(types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Overall mAP – Pretrained vs. Finetuned per Input Type",
                 fontsize=TITLE_FS, fontweight="bold", pad=14)

    bars_pre = ax.bar(x - width / 2, pre_vals, width, label="Pretrained",
                      color=COLORS["pretrained"], edgecolor="white")
    bars_ft  = ax.bar(x + width / 2, ft_vals,  width, label="Finetuned",
                      color=COLORS["finetuned"], edgecolor="white")

    # Value labels + delta boxes
    for bar, val in zip(bars_pre, pre_vals):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{val:.1f}", ha="center", va="bottom", fontsize=BAR_FS, fontweight="bold", color="black")

    for bar, val, pval in zip(bars_ft, ft_vals, pre_vals):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{val:.1f}", ha="center", va="bottom", fontsize=BAR_FS, fontweight="bold", color="black")
        _annotate_delta(ax, bar, val - pval)

    ax.set_xticks(x)
    ax.set_xticklabels([PROMPT_LABELS[t] for t in types], fontsize=TICK_FS + 2)
    ax.set_ylim(0, 125)
    ax.set_ylabel("mAP (%)", fontsize=LABEL_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=LEGEND_FS, frameon=False)

    plt.tight_layout()
    _save(fig, output_dir, "input_types_comparison.png")


# ---------------------------------------------------------------------------
# Plot 2 – Mix model: pre vs ft for each sub-type, same style as above
# ---------------------------------------------------------------------------

def plot_mix_pretrained_vs_finetuned(
    finetuned_models: dict,
    pretrained_evals: dict,
    output_dir: str,
):
    """
    For the mix model show pretrained vs. finetuned mAP for each sub-prompt
    type (bbox / point / text) with delta boxes on finetuned bars.
    """
    if "mix" not in finetuned_models or "mix" not in pretrained_evals:
        print("[SKIP] plot_mix_pretrained_vs_finetuned: mix data not available")
        return

    ft_metrics  = finetuned_models["mix"]["metrics"]
    pre_metrics = pretrained_evals["mix"]
    key         = "overall/AP_segm"
    sub_types   = ["bbox", "point", "text"]

    pre_vals = [_get_mix(pre_metrics, st, key) * 100 for st in sub_types]
    ft_vals  = [_get_mix(ft_metrics,  st, key) * 100 for st in sub_types]

    x     = np.arange(len(sub_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Mix Model – mAP per Sub-Prompt Type (Pre vs. Finetuned)",
                 fontsize=TITLE_FS, fontweight="bold", pad=14)

    bars_pre = ax.bar(x - width / 2, pre_vals, width, label="Pretrained",
                      color=COLORS["pretrained"], edgecolor="white")
    bars_ft  = ax.bar(x + width / 2, ft_vals,  width, label="Finetuned",
                      color=COLORS["finetuned"], edgecolor="white")

    for bar, val in zip(bars_pre, pre_vals):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{val:.1f}", ha="center", va="bottom", fontsize=BAR_FS, color="black")

    for bar, val, pval in zip(bars_ft, ft_vals, pre_vals):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{val:.1f}", ha="center", va="bottom", fontsize=BAR_FS, fontweight="bold", color="black")
        _annotate_delta(ax, bar, val - pval)

    ax.set_xticks(x)
    ax.set_xticklabels([PROMPT_LABELS[st] for st in sub_types], fontsize=TICK_FS + 2)
    ax.set_ylim(0, 125)
    ax.set_ylabel("mAP (%)", fontsize=LABEL_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=LEGEND_FS, frameon=False)

    plt.tight_layout()
    _save(fig, output_dir, "mix_pretrained_vs_finetuned.png")


# ---------------------------------------------------------------------------
# Plot 3 – Radar: all finetuned prompt types, without AP50/75, add med/large
# ---------------------------------------------------------------------------

def plot_finetuned_radar(
    finetuned_models: dict,
    pretrained_evals: dict,
    output_dir: str,
):
    """
    Radar chart for all finetuned models across:
    mAP, mAP_small, mAP_med, mAP_large, Car mAP, Person mAP
    (AP50 and AP75 removed per request)
    """
    radar_keys = [
        "overall/AP_segm",
        "overall/AP_small_segm",
        "overall/AP_medium_segm",
        "overall/AP_large_segm",
        "car/AP_segm",
        "person/AP_segm",
    ]
    radar_labels = ["mAP", "mAP_small", "mAP_med", "mAP_large", "Car AP", "Person AP"]

    def get_vals(pt):
        m = finetuned_models[pt]["metrics"]
        if pt == "mix":
            return [_mix_ft_avg(m, k) * 100 for k in radar_keys]
        return [_get(m, k) * 100 for k in radar_keys]

    n      = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    ax.set_title("Finetuned Models – Radar Comparison",
                 fontsize=TITLE_FS, fontweight="bold", pad=35)

    for pt in [p for p in PROMPT_TYPES if p in finetuned_models]:
        vals = get_vals(pt) + [get_vals(pt)[0]]
        ax.plot(angles, vals, linewidth=2.5, color=COLORS[pt], label=PROMPT_LABELS[pt])
        ax.fill(angles, vals, alpha=0.10, color=COLORS[pt])

    # Extra padding between the outermost ring and the axis labels
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=LABEL_FS)
    ax.tick_params(axis="x", pad=18)   # push labels outward to avoid overlap
    ax.set_ylim(0, 120)                # extend radial range so lines don't reach labels
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=TICK_FS - 2, color="grey")
    ax.spines["polar"].set_visible(False)
    # Legend moved lower to avoid title overlap
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.02),
              frameon=False, fontsize=LEGEND_FS)

    plt.tight_layout()
    _save(fig, output_dir, "finetuned_radar_comparison.png")


# ---------------------------------------------------------------------------
# Plot 4 – mAP by size heatmap (all 4 models × pre/ft × small/med/large)
# ---------------------------------------------------------------------------

def plot_size_heatmap(
    finetuned_models: dict,
    pretrained_evals: dict,
    output_dir: str,
):
    """
    Heatmap: rows = (Pre-BBox, FT-BBox, Pre-Point, FT-Point, ..., Pre-Mix, FT-Mix)
             cols = mAP_small, mAP_med, mAP_large
    """
    size_keys   = ["overall/AP_small_segm", "overall/AP_medium_segm", "overall/AP_large_segm"]
    size_labels = ["mAP_small", "mAP_med", "mAP_large"]

    row_labels = []
    matrix     = []

    for pt in PROMPT_TYPES:
        label = PROMPT_LABELS[pt]

        # --- pretrained row ---
        pre_m = pretrained_evals.get(pt, {})
        if pt == "mix":
            pre_row = [_mix_pre_avg(pre_m, k) * 100 for k in size_keys]
        else:
            pre_row = [_get(pre_m, k) * 100 for k in size_keys]
        row_labels.append(f"Pre – {label}")
        matrix.append(pre_row)

        # --- finetuned row ---
        ft_m = finetuned_models.get(pt, {}).get("metrics", {})
        if pt == "mix":
            ft_row = [_mix_ft_avg(ft_m, k) * 100 for k in size_keys]
        else:
            ft_row = [_get(ft_m, k) * 100 for k in size_keys]
        row_labels.append(f"FT – {label}")
        matrix.append(ft_row)

    data = np.array(matrix)
    fig, ax = plt.subplots(figsize=(12, len(row_labels) * 0.95 + 2.5))

    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_title("mAP by Object Size – All Models (Pre & Finetuned)",
                 fontsize=TITLE_FS, fontweight="bold", pad=14)

    ax.set_xticks(np.arange(len(size_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(size_labels, fontsize=HEAT_FS)
    ax.set_yticklabels(row_labels, fontsize=HEAT_FS)

    # Horizontal separators between pre/ft pairs
    for i in range(1, len(PROMPT_TYPES)):
        ax.axhline(i * 2 - 0.5, color="white", linewidth=3.0)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=ANNOT_FS, fontweight="bold", color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("mAP (%)", rotation=-90, va="bottom", labelpad=14, fontsize=LABEL_FS)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=HEAT_FS)

    plt.tight_layout()
    _save(fig, output_dir, "size_heatmap_all_models.png")


# ---------------------------------------------------------------------------
# Plot 5 – mAP per class heatmap (all 4 models × pre/ft × car/person)
# ---------------------------------------------------------------------------

def plot_class_heatmap(
    finetuned_models: dict,
    pretrained_evals: dict,
    output_dir: str,
):
    """
    Heatmap: rows = (Pre-BBox, FT-BBox, ..., Pre-Mix, FT-Mix)
             cols = Car mAP, Person mAP
    """
    class_keys   = ["car/AP_segm", "person/AP_segm"]
    class_labels = ["Car AP", "Person AP"]

    row_labels = []
    matrix     = []

    for pt in PROMPT_TYPES:
        label = PROMPT_LABELS[pt]

        pre_m = pretrained_evals.get(pt, {})
        if pt == "mix":
            pre_row = [_mix_pre_avg(pre_m, k) * 100 for k in class_keys]
        else:
            pre_row = [_get(pre_m, k) * 100 for k in class_keys]
        row_labels.append(f"Pre – {label}")
        matrix.append(pre_row)

        ft_m = finetuned_models.get(pt, {}).get("metrics", {})
        if pt == "mix":
            ft_row = [_mix_ft_avg(ft_m, k) * 100 for k in class_keys]
        else:
            ft_row = [_get(ft_m, k) * 100 for k in class_keys]
        row_labels.append(f"FT – {label}")
        matrix.append(ft_row)

    data = np.array(matrix)
    fig, ax = plt.subplots(figsize=(9, len(row_labels) * 0.95 + 2.5))

    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_title("AP per Class – All Models (Pre & Finetuned)",
                 fontsize=TITLE_FS, fontweight="bold", pad=14)

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(class_labels, fontsize=HEAT_FS)
    ax.set_yticklabels(row_labels, fontsize=HEAT_FS)

    for i in range(1, len(PROMPT_TYPES)):
        ax.axhline(i * 2 - 0.5, color="white", linewidth=3.0)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=ANNOT_FS, fontweight="bold", color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.set_ylabel("AP (%)", rotation=-90, va="bottom", labelpad=14, fontsize=LABEL_FS)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=HEAT_FS)

    plt.tight_layout()
    _save(fig, output_dir, "class_heatmap_all_models.png")


# ---------------------------------------------------------------------------
# Plot 6 – Improvement heatmap (ΔmAP after finetuning)
# ---------------------------------------------------------------------------

def plot_improvement_heatmap(
    finetuned_models: dict,
    pretrained_evals: dict,
    output_dir: str,
):
    """
    Heatmap of ΔmAP (finetuned - pretrained) for bbox / point / text.
    Rows = prompt types, Columns = metrics.
    """
    types_to_plot = [pt for pt in ["bbox", "point", "text"]
                     if pt in finetuned_models and pt in pretrained_evals]
    metric_keys = [
        "overall/AP_segm",
        "overall/AP_small_segm",
        "overall/AP_medium_segm",
        "overall/AP_large_segm",
        "car/AP_segm",
        "person/AP_segm",
    ]
    metric_labels = ["mAP", "mAP_small", "mAP_med", "mAP_large", "Car mAP", "Person mAP"]

    matrix = []
    row_labels = []
    for pt in types_to_plot:
        row = [
            (_get(finetuned_models[pt]["metrics"], mk) - _get(pretrained_evals[pt], mk)) * 100
            for mk in metric_keys
        ]
        matrix.append(row)
        row_labels.append(PROMPT_LABELS[pt])

    data    = np.array(matrix)
    abs_max = max(abs(data.min()), abs(data.max()), 1.0)

    fig, ax = plt.subplots(figsize=(13, 3 + len(types_to_plot) * 0.9))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-abs_max, vmax=abs_max, aspect="auto")
    ax.set_title("ΔmAP Improvement Heatmap (Finetuned − Pretrained)",
                 fontsize=TITLE_FS, fontweight="bold", pad=14)

    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(metric_labels, fontsize=TICK_FS)
    ax.set_yticklabels(row_labels, fontsize=TICK_FS, fontweight="bold")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val  = data[i, j]
            sign = "+" if val >= 0 else ""
            ax.text(j, i, f"{sign}{val:.1f}", ha="center", va="center",
                    fontsize=ANNOT_FS, fontweight="bold", color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.set_ylabel("ΔmAP (%)", rotation=-90, va="bottom", labelpad=14, fontsize=LABEL_FS)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=TICK_FS)

    plt.tight_layout()
    _save(fig, output_dir, "improvement_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 7 – Augmentation strategy comparison (sam_aug sweep)
# ---------------------------------------------------------------------------

def _load_aug_experiments(aug_dir: str) -> list:
    """
    Load all experiments in aug_dir. Each sub-folder is expected to have
    best_metrics.json and config.yaml.

    Returns a list of dicts sorted by overall/AP_segm descending:
        [{'label': str, 'aug_strategy': str, 'metrics': dict}, ...]
    """
    entries = []
    base = Path(aug_dir)
    if not base.exists():
        print(f"[WARNING] Aug directory not found: {aug_dir}")
        return entries

    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_file = run_dir / "best_metrics.json"
        config_file  = run_dir / "config.yaml"
        if not metrics_file.exists() or not config_file.exists():
            print(f"[SKIP] {run_dir.name}: missing files")
            continue
        with open(metrics_file) as f:
            metrics = json.load(f)
        with open(config_file) as f:
            config = yaml.safe_load(f)
        aug = config.get("training", {}).get("aug_strategy",
              config.get("experiment_name", run_dir.name))
        # Pretty-print: underscores to spaces, title-case
        label = aug.replace("_", " ").title()
        entries.append({"label": label, "aug_strategy": aug, "metrics": metrics})

    # Sort best → worst by overall mAP
    entries.sort(key=lambda e: e["metrics"].get("overall/AP_segm", 0), reverse=True)
    return entries


# Assign a distinct colour to each augmentation strategy
_AUG_PALETTE = [
    "#1f4e79", "#2d9e6b", "#e86c1f", "#7b2d8b",
    "#c0392b", "#f39c12", "#16a085", "#8e44ad",
]


def plot_aug_comparison(aug_dir: str, output_dir: str):
    """
    Two plots saved for the augmentation sweep:
      1. aug_strategy_mAP.png  – horizontal bar chart ranked by mAP,
                                  with Car AP and Person AP as inner bars.
      2. aug_strategy_heatmap.png – heatmap (strategies × metrics).
    """
    entries = _load_aug_experiments(aug_dir)
    if not entries:
        print("[SKIP] plot_aug_comparison: no data found")
        return

    labels   = [e["label"]      for e in entries]
    map_vals = [e["metrics"].get("overall/AP_segm", 0) * 100  for e in entries]
    car_vals = [e["metrics"].get("car/AP_segm",     0) * 100  for e in entries]
    ped_vals = [e["metrics"].get("person/AP_segm",  0) * 100  for e in entries]
    colors   = [_AUG_PALETTE[i % len(_AUG_PALETTE)] for i in range(len(entries))]

    # ---- Plot A: simple vertical bar chart – mAP per strategy ---------------
    n = len(labels)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.6), 7))
    ax.set_title("Augmentation Strategy Comparison – mAP",
                 fontsize=TITLE_FS, fontweight="bold", pad=14)

    bars = ax.bar(x, map_vals, color=COLORS["finetuned"], edgecolor="white", width=0.6)

    for bar, val in zip(bars, map_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=BAR_FS, fontweight="bold", color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=HEAT_FS, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("mAP (%)", fontsize=LABEL_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, output_dir, "aug_strategy_mAP.png")

    # ---- Plot B: heatmap (strategies × 6 metrics) ---------------------------
    hmap_keys = [
        "overall/AP_segm",
        "overall/AP_small_segm",
        "overall/AP_medium_segm",
        "overall/AP_large_segm",
        "car/AP_segm",
        "person/AP_segm",
    ]
    hmap_labels = ["mAP", "mAP_small", "mAP_med", "mAP_large", "Car AP", "Person AP"]

    matrix = np.array([
        [e["metrics"].get(k, 0) * 100 for k in hmap_keys]
        for e in entries
    ])

    fig2, ax2 = plt.subplots(figsize=(14, max(4, n * 0.85 + 1.5)))
    im = ax2.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax2.set_title("Augmentation Strategy – Full Metrics Heatmap",
                  fontsize=TITLE_FS, fontweight="bold", pad=14)

    ax2.set_xticks(np.arange(len(hmap_labels)))
    ax2.set_yticks(np.arange(n))
    ax2.set_xticklabels(hmap_labels, fontsize=HEAT_FS)
    ax2.set_yticklabels(labels, fontsize=HEAT_FS)
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax2.text(j, i, f"{matrix[i, j]:.1f}",
                     ha="center", va="center",
                     fontsize=ANNOT_FS, fontweight="bold", color="black")

    cbar = fig2.colorbar(im, ax=ax2, fraction=0.015, pad=0.02)
    cbar.ax.set_ylabel("(%)", rotation=-90, va="bottom", labelpad=12, fontsize=LABEL_FS)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=HEAT_FS)

    plt.tight_layout()
    _save(fig2, output_dir, "aug_strategy_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 8 – Generalization: Dev vs. Validation splits (Pre vs. Finetuned)
# ---------------------------------------------------------------------------

def plot_generalization_comparison(finetuned_models, pretrained_evals, aug_dir, output_dir):
    """
    Shows performance on Dev vs Validation splits to demonstrate generalization.
    Points to compare:
    1. Pretrained Dev: results_eval/eval_sam_metrics_dev.json
    2. Finetuned Dev: results_finetune/sam_aug/sam_no_aug_tqbwlssf/best_metrics.json
    3. Pretrained Val: results_eval/eval_sam_metrics_validation.json
    4. Finetuned Val: results_finetune/final_finetuned/sam_bbox_j6cstc09/best_metrics.json
    """
    # 1. Load specific metrics
    def load_json(path):
        p = Path(path)
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    # Hardcoded paths based on the requested specific models/splits
    pre_dev_path = Path(output_dir).parent / "eval_sam_metrics_dev.json"
    ft_dev_path  = Path(aug_dir) / "sam_no_aug_tqbwlssf" / "best_metrics.json"
    pre_val_path = Path(output_dir).parent / "eval_sam_metrics_validation.json"
    # Find the bbox folder in finetuned_dir (passed as finetuned_models keys)
    # We'll look for j6cstc09 specifically as identified
    ft_val_base = Path(aug_dir).parent / "final_finetuned"
    ft_val_path = ft_val_base / "sam_bbox_j6cstc09" / "best_metrics.json"

    m_pre_dev = load_json(pre_dev_path)
    m_ft_dev  = load_json(ft_dev_path)
    m_pre_val = load_json(pre_val_path)
    m_ft_val  = load_json(ft_val_path)

    if not (m_pre_dev and m_ft_dev and m_pre_val and m_ft_val):
        print(f"[SKIP] plot_generalization_comparison: missing one or more metric files.")
        if not m_pre_dev: print(f"  Missing: {pre_dev_path}")
        if not m_ft_dev:  print(f"  Missing: {ft_dev_path}")
        if not m_pre_val: print(f"  Missing: {pre_val_path}")
        if not m_ft_val:  print(f"  Missing: {ft_val_path}")
        return

    splits = ["Dev Split", "Validation Split"]
    pre_vals = [
        m_pre_dev.get("overall/AP_segm", 0) * 100,
        m_pre_val.get("overall/AP_segm", 0) * 100
    ]
    ft_vals = [
        m_ft_dev.get("overall/AP_segm", 0) * 100,
        m_ft_val.get("overall/AP_segm", 0) * 100
    ]

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Finetuning Generalization – Dev vs. Validation Splits",
                 fontsize=TITLE_FS, fontweight="bold", pad=20)

    bars_pre = ax.bar(x - width/2, pre_vals, width, label="Pretrained",
                      color=COLORS["pretrained"], edgecolor="white")
    bars_ft  = ax.bar(x + width/2, ft_vals,  width, label="Finetuned",
                      color=COLORS["finetuned"], edgecolor="white")

    # Labels
    for bar, val in zip(bars_pre, pre_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}", ha="center", va="bottom", fontsize=BAR_FS, fontweight="bold", color="black")

    for bar, val, pval in zip(bars_ft, ft_vals, pre_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}", ha="center", va="bottom", fontsize=BAR_FS, fontweight="bold", color="black")
        _annotate_delta(ax, bar, val - pval)

    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=HEAT_FS)
    ax.set_ylim(0, 125)
    ax.set_ylabel("mAP (%)", fontsize=LABEL_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="upper left")

    plt.tight_layout()
    _save(fig, output_dir, "generalization_comparison.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for Week2 SAM fine-tuning results."
    )
    parser.add_argument(
        "--finetuned_dir",
        type=str,
        default="/ghome/group01/C5/benet/C5-Team1/Week2/results_finetune/final_finetuned",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="/ghome/group01/C5/benet/C5-Team1/Week2/results_eval",
    )
    parser.add_argument(
        "--aug_dir",
        type=str,
        default="/ghome/group01/C5/benet/C5-Team1/Week2/results_finetune/sam_aug",
        help="Directory containing augmentation sweep experiments.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ghome/group01/C5/benet/C5-Team1/Week2/results_eval/plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading finetuned models...")
    finetuned_models = load_finetuned_models(args.finetuned_dir)
    print(f"  Found: {list(finetuned_models.keys())}")

    print("Loading pretrained eval metrics...")
    pretrained_evals = load_pretrained_evals(args.eval_dir)
    print(f"  Found: {list(pretrained_evals.keys())}")

    print(f"\nSaving plots to: {args.output_dir}")

    # 1. All input types mAP bar + delta boxes
    plot_input_types_comparison(finetuned_models, pretrained_evals, args.output_dir)

    # 2. Mix sub-types mAP bar + delta boxes
    plot_mix_pretrained_vs_finetuned(finetuned_models, pretrained_evals, args.output_dir)

    # 3. Radar (no AP50/AP75, added mAP_med / mAP_large)
    plot_finetuned_radar(finetuned_models, pretrained_evals, args.output_dir)

    # 4. Size heatmap (all 4 models, pre & ft)
    plot_size_heatmap(finetuned_models, pretrained_evals, args.output_dir)

    # 5. Class heatmap (all 4 models, pre & ft)
    plot_class_heatmap(finetuned_models, pretrained_evals, args.output_dir)

    # 6. Delta improvement heatmap
    plot_improvement_heatmap(finetuned_models, pretrained_evals, args.output_dir)

    # 7. Augmentation strategy comparison (sam_aug sweep)
    print("\nGenerating augmentation strategy plots...")
    plot_aug_comparison(args.aug_dir, args.output_dir)

    # 8. Generalization comparison
    print("\nGenerating generalization comparison plot...")
    plot_generalization_comparison(finetuned_models, pretrained_evals, args.aug_dir, args.output_dir)

    print("\nDone! All plots saved.")


if __name__ == "__main__":
    main()
