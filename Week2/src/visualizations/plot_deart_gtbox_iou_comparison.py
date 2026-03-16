from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Palette borrowed from the provided notebook style.
COLOR_BEFORE = "#1D95B7" # light gray
COLOR_AFTER = "#214972"   # deep blue
COLOR_POS = "#2ca02c"
COLOR_NEG = "#d62728"


def _load_metrics(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def _get_class_iou(metrics: Dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cls, values in metrics.get("per_class", {}).items():
        out[cls] = float(values.get("mask_box_iou/mean", 0.0))
    return out


def plot_overall_iou(pretrained: Dict, finetuned: Dict, out_path: Path) -> None:
    labels = ["Pretrained SAM", "Finetuned SAM"]
    values = [
        float(pretrained.get("mask_box_iou/mean", 0.0)),
        float(finetuned.get("mask_box_iou/mean", 0.0)),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=[COLOR_BEFORE, COLOR_AFTER], width=0.55)

    ax.set_ylabel("mask_box_iou", fontsize=14)
    ax.set_title("DeART GT-Box Prompt: Overall mask_box_iou", fontsize=16, fontweight="bold", pad=16)
    ax.set_ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    delta = values[0] - values[1]
    ax.text(
        0.5,
        0.94,
        f"Delta (Pretrained - Finetuned): {delta:+.4f}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLOR_POS if delta >= 0 else COLOR_NEG,
    )

    for b in bars:
        h = b.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_biggest_class_gains(pretrained: Dict, finetuned: Dict, out_path: Path) -> None:
    pre_cls = _get_class_iou(pretrained)
    fin_cls = _get_class_iou(finetuned)

    all_classes = sorted(set(pre_cls) | set(fin_cls))
    gains: List[Tuple[str, float]] = []
    for cls in all_classes:
        gains.append((cls, pre_cls.get(cls, 0.0) - fin_cls.get(cls, 0.0)))

    gains.sort(key=lambda x: x[1], reverse=True)
    classes = [g[0] for g in gains]
    values = [g[1] for g in gains]
    colors = [COLOR_POS if v >= 0 else COLOR_NEG for v in values]

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(classes))
    bars = ax.barh(y, values, color=colors)

    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("IoU gain (Pretrained - Finetuned)", fontsize=13)
    ax.set_title("Biggest mask_box_iou Gains Per Class", fontsize=16, fontweight="bold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for b, v in zip(bars, values):
        x = b.get_width()
        offset = 0.005 if v >= 0 else -0.005
        ha = "left" if v >= 0 else "right"
        ax.text(x + offset, b.get_y() + b.get_height() / 2, f"{v:+.4f}", va="center", ha=ha, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent.parent
    base = root / "results_deart_NOCLASSMAP_NEWVIZ"

    p = argparse.ArgumentParser(description="Plot DeART GT-box IoU comparison (pretrained vs finetuned SAM).")
    p.add_argument(
        "--pretrained_metrics",
        type=str,
        default=str(base / "deart_gt_box_pretrained_validation" / "metrics.json"),
    )
    p.add_argument(
        "--finetuned_metrics",
        type=str,
        default=str(base / "deart_gt_box_finetuned_validation" / "metrics.json"),
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(base / "slide_assets"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pre_path = Path(args.pretrained_metrics)
    fin_path = Path(args.finetuned_metrics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrained = _load_metrics(pre_path)
    finetuned = _load_metrics(fin_path)

    out_overall = out_dir / "gtbox_iou_overall_pretrained_vs_finetuned.png"
    out_class = out_dir / "gtbox_iou_biggest_class_gains.png"

    plot_overall_iou(pretrained, finetuned, out_overall)
    plot_biggest_class_gains(pretrained, finetuned, out_class)

    print("Saved:")
    print(f"  - {out_overall}")
    print(f"  - {out_class}")


if __name__ == "__main__":
    main()
