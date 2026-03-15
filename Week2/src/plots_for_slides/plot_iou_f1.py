#!/usr/bin/env python
"""
Plot IoU threshold vs F1-score curves comparing pretrained and finetuned SAM
models across all input types (bbox, point, text).

Two plots are saved:
  - iou_vs_f1_per_prompt.png   : one subplot per input type
  - iou_vs_f1_combined.png     : all input types on a single axes

Usage:
    python -m src.plots_for_slides.plot_iou_f1 \
        --pretrained results_eval/eval_sam_metrics_mix_validation.json \
        --finetuned  results_eval/finetuned_comparison_metrics.json \
        --output     results_eval/plots/

Input JSON formats:
  pretrained : flat dict with keys like  `{prompt_type}_overall/F1_score_{iou}_segm`
               (produced by eval_sam_metrics.py --prompt_type mix)
  finetuned  : nested dict  `{ model_dir: { prompt_type: { metric: value } } }`
               (produced by batch_eval_finetuned.py)
               Only dedicated per-prompt models are used (mix directories are skipped).
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style (consistent with plot_utils.py)
# ---------------------------------------------------------------------------
TITLE_FS  = 20
LABEL_FS  = 16
TICK_FS   = 14
LEGEND_FS = 14

matplotlib.rcParams.update({
    "font.size":       LABEL_FS,
    "axes.titlesize":  TITLE_FS,
    "axes.labelsize":  LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
})

# COCOeval IoU thresholds expressed as integer percentages (50, 55, …, 95)
IOU_THRESHOLDS = list(range(50, 100, 5))
PROMPT_TYPES   = ["bbox", "point", "text"]
PROMPT_LABELS  = {"bbox": "Bbox", "point": "Point", "text": "Text"}

COLORS     = {"pretrained": "#8ecae6", "finetuned": "#023047"}
LINESTYLES = {"pretrained": "--",      "finetuned": "-"}
MARKERS    = {"pretrained": "o",       "finetuned": "s"}

# Per-prompt colour pair (finetuned, pretrained) for the combined plot
PROMPT_COLORS = {
    "bbox":  ("#1f4e79", "#8ecae6"),
    "point": ("#2d9e6b", "#a8d5c2"),
    "text":  ("#e86c1f", "#f4b49e"),
    "mix":   ("#7b2d8b", "#c9a8d4"),
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def extract_pretrained_curve(pretrained: dict, prompt_type: str) -> list[float]:
    """Extract the overall F1 curve for *prompt_type* from the pretrained dict."""
    return [
        pretrained.get(f"{prompt_type}_overall/F1_score_{iou}_segm", float("nan"))
        for iou in IOU_THRESHOLDS
    ]


def extract_finetuned_curve(finetuned: dict, prompt_type: str) -> tuple[list[float], str]:
    """
    Return the F1 curve and directory name for the dedicated per-prompt finetuned
    model with the highest mF1 score. Mix-model directories are ignored.
    """
    best_mf1, best_f1s, best_name = -1.0, [], ""
    for model_name, model_data in finetuned.items():
        if "mix" in model_name.lower() or prompt_type not in model_data:
            continue
        m   = model_data[prompt_type]
        mf1 = m.get("overall/mF1_segm", 0.0)
        if mf1 > best_mf1:
            best_mf1  = mf1
            best_f1s  = [m.get(f"overall/F1_score_{iou}_segm", float("nan")) for iou in IOU_THRESHOLDS]
            best_name = model_name
    return best_f1s, best_name


def extract_mix_curve(finetuned: dict) -> list[float]:
    """
    Return the F1 curve for the mix finetuned model, averaging across its three
    sub-prompt evaluations (bbox, point, text).
    """
    mix_models = {name: data for name, data in finetuned.items() if "mix" in name.lower()}
    if not mix_models:
        return [float("nan")] * len(IOU_THRESHOLDS)

    # Pick the mix model with the highest average mF1 across sub-prompts
    best_model = max(
        mix_models.values(),
        key=lambda d: np.nanmean([d.get(pt, {}).get("overall/mF1_segm", 0.0)
                                   for pt in ("bbox", "point", "text")]),
    )
    curves = [
        [best_model.get(pt, {}).get(f"overall/F1_score_{iou}_segm", float("nan"))
         for iou in IOU_THRESHOLDS]
        for pt in ("bbox", "point", "text")
        if pt in best_model
    ]
    return list(np.nanmean(curves, axis=0)) if curves else [float("nan")] * len(IOU_THRESHOLDS)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _style_ax(ax, ylabel: bool = False):
    ax.set_xlabel("IoU Threshold")
    if ylabel:
        ax.set_ylabel("F1-Score")
    ax.set_xlim(0.48, 0.97)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":",  alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_iou_vs_f1(pretrained: dict, finetuned: dict, output_dir: str):
    """One subplot per prompt type showing pretrained vs finetuned F1 curves.
    An additional Mix-model line (average of bbox/point/text) is overlaid on each subplot."""
    os.makedirs(output_dir, exist_ok=True)
    iou_x = [t / 100 for t in IOU_THRESHOLDS]
    mix_f1s = extract_mix_curve(finetuned)

    fig, axes = plt.subplots(
        1, len(PROMPT_TYPES),
        figsize=(7 * len(PROMPT_TYPES), 6),
        sharey=True,
    )
    fig.suptitle(
        "IoU Threshold vs F1-Score — Pretrained vs Finetuned",
        fontsize=TITLE_FS + 2, fontweight="bold", y=1.02,
    )

    for ax, prompt_type in zip(axes, PROMPT_TYPES):
        ax.plot(
            iou_x, extract_pretrained_curve(pretrained, prompt_type),
            color=COLORS["pretrained"], linestyle=LINESTYLES["pretrained"],
            marker=MARKERS["pretrained"], linewidth=2.5, markersize=7,
            label="Pretrained",
        )
        ft_f1s, _ = extract_finetuned_curve(finetuned, prompt_type)
        ax.plot(
            iou_x, ft_f1s,
            color=COLORS["finetuned"], linestyle=LINESTYLES["finetuned"],
            marker=MARKERS["finetuned"], linewidth=2.5, markersize=7,
            label="Finetuned",
        )
        ax.plot(
            iou_x, mix_f1s,
            color=PROMPT_COLORS["mix"][0], linestyle="-.",
            marker="^", linewidth=2.0, markersize=7,
            label="Finetuned (Mix, avg)",
        )
        ax.set_title(PROMPT_LABELS[prompt_type], fontweight="bold")
        _style_ax(ax, ylabel=(prompt_type == PROMPT_TYPES[0]))
        ax.legend(frameon=False)

    plt.tight_layout()
    out = os.path.join(output_dir, "iou_vs_f1_per_prompt.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_iou_vs_f1_combined(pretrained: dict, finetuned: dict, output_dir: str):
    """All prompt types on a single axes; dashed = pretrained, solid = finetuned.
    An extra Mix line (average of bbox/point/text from the mix model) is also included."""
    os.makedirs(output_dir, exist_ok=True)
    iou_x = [t / 100 for t in IOU_THRESHOLDS]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("IoU Threshold vs F1-Score — All Prompt Types", fontweight="bold")

    for prompt_type in PROMPT_TYPES:
        label             = PROMPT_LABELS[prompt_type]
        ft_col, pre_col   = PROMPT_COLORS[prompt_type]
        ft_f1s, _         = extract_finetuned_curve(finetuned, prompt_type)

        ax.plot(iou_x, extract_pretrained_curve(pretrained, prompt_type),
                color=pre_col, linestyle="--", marker="o", linewidth=2.0, markersize=6,
                label=f"{label} Pretrained")
        ax.plot(iou_x, ft_f1s,
                color=ft_col,  linestyle="-",  marker="s", linewidth=2.5, markersize=7,
                label=f"{label} Finetuned")

    # Mix finetuned model: average F1 across bbox / point / text sub-prompts
    mix_col, _ = PROMPT_COLORS["mix"]
    ax.plot(iou_x, extract_mix_curve(finetuned),
            color=mix_col, linestyle="-.", marker="^", linewidth=2.5, markersize=7,
            label="Mix Finetuned (avg)")

    _style_ax(ax, ylabel=True)
    ax.legend(fontsize=LEGEND_FS - 1, frameon=True, ncol=2, loc="upper right",
              framealpha=0.85, edgecolor="#cccccc")

    plt.tight_layout()
    out = os.path.join(output_dir, "iou_vs_f1_combined.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot IoU threshold vs F1-score curves.")
    parser.add_argument("--pretrained", required=True,
                        help="Pretrained JSON (mix evaluation with prefixed keys).")
    parser.add_argument("--finetuned",  required=True,
                        help="Finetuned comparison JSON (model_dir → prompt_type → metrics).")
    parser.add_argument("--output", default="results_eval/plots/",
                        help="Output directory for the saved plots.")
    args = parser.parse_args()

    with open(args.pretrained) as f:
        pretrained = json.load(f)
    with open(args.finetuned) as f:
        finetuned = json.load(f)

    plot_iou_vs_f1(pretrained, finetuned, args.output)
    plot_iou_vs_f1_combined(pretrained, finetuned, args.output)


if __name__ == "__main__":
    main()
