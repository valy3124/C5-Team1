from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Keep a slide-friendly palette aligned with prior notebook style.
COLOR_GT = "#D3D3D3"
COLOR_DET = "#214972"


def _load_metrics(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def _to_int(v: float) -> int:
    return int(round(float(v)))


def _extract_shared_counts(pretrained: Dict, finetuned: Dict) -> Tuple[int, int]:
    gt_pre = _to_int(pretrained.get("box_count/total_gt", 0))
    gt_fin = _to_int(finetuned.get("box_count/total_gt", 0))
    det_pre = _to_int(pretrained.get("box_count/total_detected", 0))
    det_fin = _to_int(finetuned.get("box_count/total_detected", 0))

    # If values differ slightly, fall back to pretrained values for plotting.
    if gt_pre != gt_fin or det_pre != det_fin:
        print("Warning: pretrained/finetuned box counts differ; plotting pretrained values.")

    return gt_pre, det_pre


def plot_boxcount_single(pretrained: Dict, finetuned: Dict, out_path: Path) -> None:
    gt_count, det_count = _extract_shared_counts(pretrained, finetuned)

    labels = ["Ground Truth", "Detected"]
    values = [gt_count, det_count]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=[COLOR_GT, COLOR_DET], width=0.55)

    ax.set_ylabel("Number of Boxes", fontsize=13)
    ax.set_title("GroundingDINO Box Counts on DeART", fontsize=16, fontweight="bold", pad=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymax = max(values)
    ax.set_ylim(0, ymax * 1.20)

    for b in bars:
        h = b.get_height()
        ax.annotate(
            f"{int(h)}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    delta = det_count - gt_count
    ax.text(
        0.5,
        ymax * 1.10,
        f"Over-detection: {delta:+d} boxes",
        ha="center",
        va="center",
        fontsize=12,
        color="#2ca02c" if delta >= 0 else "#d62728",
        fontweight="bold",
        transform=ax.get_yaxis_transform(),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _most_overdetected_classes(metrics: Dict, top_k: int = 3) -> List[Tuple[str, int, int, int, float]]:
    rows: List[Tuple[str, int, int, int, float]] = []
    per_class = metrics.get("per_class", {})

    for cls, vals in per_class.items():
        n_gt = int(vals.get("n_gt", 0))
        n_det = int(vals.get("n_det", 0))
        over = n_det - n_gt
        if over <= 0:
            continue
        ratio = float(n_det) / float(n_gt) if n_gt > 0 else float("inf")
        rows.append((cls, over, n_det, n_gt, ratio))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]


def plot_most_overdetected_classes(metrics: Dict, out_path: Path, top_k: int = 3) -> None:
    rows = _most_overdetected_classes(metrics, top_k=top_k)
    if not rows:
        print("No over-detected classes found; skipping class plot.")
        return

    classes = [r[0] for r in rows]
    over_counts = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    x = np.arange(len(classes))
    bars = ax.bar(x, over_counts, color=COLOR_DET, width=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel("Over-detected boxes (n_det - n_gt)", fontsize=13)
    ax.set_title("Top-3 Most Over-Detected Classes (GroundingDINO)", fontsize=16, fontweight="bold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for b, row in zip(bars, rows):
        _cls, over, n_det, n_gt, ratio = row
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(over_counts) * 0.02,
            f"+{over} (det={n_det}, gt={n_gt}, x{ratio:.1f})",
            va="bottom",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent.parent
    base = root / "results_deart_NOCLASSMAP_NEWVIZ"

    p = argparse.ArgumentParser(description="Plot GroundingDINO detected vs GT box counts for pretrained/finetuned text-prompt SAM.")
    p.add_argument(
        "--pretrained_metrics",
        type=str,
        default=str(base / "deart_text_prompt_pretrained_validation" / "metrics.json"),
    )
    p.add_argument(
        "--finetuned_metrics",
        type=str,
        default=str(base / "deart_text_prompt_finetuned_validation" / "metrics.json"),
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

    out_path_counts = out_dir / "text_prompt_gt_vs_detected_box_counts.png"
    out_path_classes = out_dir / "text_prompt_most_overdetected_classes.png"

    plot_boxcount_single(pretrained, finetuned, out_path_counts)
    # Class-level n_det/n_gt is identical between models here, so one file is enough.
    plot_most_overdetected_classes(pretrained, out_path_classes, top_k=3)

    print("Saved:")
    print(f"  - {out_path_counts}")
    print(f"  - {out_path_classes}")


if __name__ == "__main__":
    main()
