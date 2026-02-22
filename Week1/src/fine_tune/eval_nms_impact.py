import argparse
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Add Week1 to sys.path so `src` can be imported
script_dir = Path(__file__).resolve().parent
week_dir = script_dir.parent.parent
sys.path.append(str(week_dir))

from src.datasets import KITTIMOTS
from src.models.faster_rcnn import FasterRCNNModel
from src.fine_tune.fine_tune import (
    get_transforms, evaluate, collate_fn,
    KITTIMOTSToTorchvision, ApplyAlbumentations,
)
from src.inference.evaluation import CocoMetrics


# Class colors: 1=Car (blue), 2=Pedestrian (red)
CLASS_COLORS = {1: "red", 2: "blue"}
CLASS_NAMES  = {1: "Pedestrian", 2: "Car"}


class MockExp:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

class MockRun:
    def __init__(self, m):
        self.model = m


def draw_boxes(ax, boxes, labels, scores=None, color_map=CLASS_COLORS):
    """Draw bounding boxes on a matplotlib axis."""
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        color = color_map.get(int(label), "yellow")
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        if scores is not None:
            ax.text(x1, y1 - 2, f"{scores[i]:.2f}", color=color, fontsize=6, va='bottom')


def plot_comparisons(image, gt_anns, predictions_dict, output_path, conf_threshold=0.4):
    """
    Plots the ground truth and multiple NMS threshold predictions side-by-side
    and saves the figure to output_path.
    """
    num_subplots = 1 + len(predictions_dict)
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 4))

    # 1. Ground Truth
    axes[0].imshow(image)
    gt_boxes  = np.array([ann.bbox_xyxy for ann in gt_anns])
    gt_labels = np.array([ann.class_id   for ann in gt_anns])
    if len(gt_boxes) > 0:
        draw_boxes(axes[0], gt_boxes, gt_labels)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    # 2. Each NMS threshold prediction
    for ax_idx, (iou_thresh, pred) in enumerate(predictions_dict.items(), start=1):
        axes[ax_idx].imshow(image)

        valid = pred['scores'] > conf_threshold
        boxes  = pred['bboxes_xyxy'][valid]
        labels = pred['category_ids'][valid]
        scores = pred['scores'][valid]

        if len(boxes) > 0:
            draw_boxes(axes[ax_idx], boxes, labels, scores=scores)

        axes[ax_idx].set_title(f"NMS={iou_thresh} ({valid.sum()} boxes)")
        axes[ax_idx].axis("off")

    plt.suptitle(f"Confidence threshold: {conf_threshold}", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",    type=str,
                        default="/export/home/group01/mcv/datasets/C5/KITTI-MOTS/",
                        help="Path to KITTI-MOTS dataset")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Path to the experiment directory")
    parser.add_argument("--conf",    type=float, default=0.4,
                        help="Confidence threshold for visualizations")
    args = parser.parse_args()

    exp_dir      = Path(args.exp_dir)
    config_path  = exp_dir / "config.yaml"
    weights_path = exp_dir / "best_model.pth"
    output_dir   = exp_dir / "nms_evaluation"
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ----- Dataset -----
    print("Loading Validation Dataset...")
    val_base = KITTIMOTS(root=args.root, split="dev", seed=42, split_ratio=0.8)
    val_ds   = ApplyAlbumentations(
        KITTIMOTSToTorchvision(val_base),
        tf=get_transforms(is_train=False)
    )
    val_loader  = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    metrics_obj = CocoMetrics(root=args.root, split="dev", ann_source="txt", seed=42, split_ratio=0.8)

    num_classes     = 3  # background + Car + Pedestrian
    freeze_strategy = cfg['training'].get('freeze_strategy', 1)

    # ----- Sweep -----
    iou_thresholds   = [0.3, 0.5, 0.7, 0.9]
    indices_to_test  = [100, 300, 600, 800, 1000, 1500]
    metrics_results  = {}
    saved_predictions = {idx: {} for idx in indices_to_test}

    for iou in iou_thresholds:
        print(f"\n=================================\nEvaluating NMS Threshold: {iou}\n=================================")

        model = FasterRCNNModel(weights=None, device=str(device), iou=iou)
        model.prepare_finetune(num_classes=num_classes, freeze_strategy=freeze_strategy)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        exp = MockExp(cfg, device)
        run = MockRun(m=model)
        eval_result = evaluate(exp, run, val_loader, metrics_obj)
        map_score   = eval_result.metrics.get('overall/AP', 0.0)
        print(f"-> mAP for NMS {iou}: {map_score:.4f}")
        metrics_results[str(iou)] = eval_result.metrics   # JSON keys must be str

        # Grab raw predictions for qualitative plots
        for idx in indices_to_test:
            try:
                image, anns, meta = val_base[idx]
                saved_predictions[idx][iou] = model.predict(image)
            except IndexError:
                pass

    # ----- Save JSON -----
    metrics_path = output_dir / "nms_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_results, f, indent=4)
    print(f"\nSaved metrics to {metrics_path}")

    # Pretty summary table
    print("\n{:>6} | {:>10} | {:>10} | {:>10}".format(
        "NMS", "AP", "AP_50", "AP_75"))
    print("-" * 44)
    for iou_str, m in metrics_results.items():
        print("{:>6} | {:>10.4f} | {:>10.4f} | {:>10.4f}".format(
            iou_str,
            m.get('overall/AP', 0),
            m.get('overall/AP_50', 0),
            m.get('overall/AP_75', 0),
        ))

    # ----- Save plots -----
    print("\nGenerating side-by-side plots...")
    for idx in indices_to_test:
        try:
            image, anns, meta = val_base[idx]
            seq      = meta['seq']
            frame_id = meta['frame']

            out_path = output_dir / f"compare_seq{seq}_frame{frame_id}.png"
            plot_comparisons(image, anns, saved_predictions[idx], out_path,
                             conf_threshold=args.conf)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  Failed for index {idx}: {e}")

    print("\nEvaluation Complete.")


if __name__ == "__main__":
    main()
