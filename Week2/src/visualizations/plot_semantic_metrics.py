import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Define paths
ROOT_DIR = Path("/ghome/group01/C5/vali/C5-Team1/Week2")
pretrained_path = ROOT_DIR / "results_semantic" / "semantic_text_prompt_validation" / "metrics.json"
finetuned_path = ROOT_DIR / "results_semantic" / "semantic_finetuned_sam_validation_new" / "metrics.json"
out_dir = ROOT_DIR / "results_semantic" / "plots"
out_dir.mkdir(exist_ok=True, parents=True)

# Load data
with open(pretrained_path, "r") as f:
    pre_metrics = json.load(f)
with open(finetuned_path, "r") as f:
    ft_metrics = json.load(f)

# Define metrics to plot
metrics_to_plot = [
    ("overall/mIoU", "mIoU"),
    ("person/IoU", "Person IoU"),
    ("car/IoU", "Car IoU"),
]

labels = [m[1] for m in metrics_to_plot]
pre_vals = [pre_metrics[m[0]] for m in metrics_to_plot]
ft_vals = [ft_metrics[m[0]] for m in metrics_to_plot]

x = np.arange(len(labels))
width = 0.35

# High-quality aesthetic settings
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, pre_vals, width, label='Pretrained + Text Prompt', color='#3498db')
rects2 = ax.bar(x + width/2, ft_vals, width, label='Finetuned + BBox Prompt', color='#2ecc71')

ax.set_ylabel('Score')
ax.set_title('Semantic Segmentation Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper left')

# Add values on top of bars
def autolabel_float(rects, ax_obj):
    for rect in rects:
        height = rect.get_height()
        ax_obj.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

def autolabel_int(rects, ax_obj):
    for rect in rects:
        height = rect.get_height()
        ax_obj.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel_float(rects1, ax)
autolabel_float(rects2, ax)

fig.tight_layout()
plt.savefig(out_dir / "accuracy_comparison.png", dpi=300)
plt.close()

# Plot Performance Metrics (FPS and Latency separately)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# FPS
fps_labels = ['FPS']
pre_fps = [pre_metrics['performance/avg_fps']]
ft_fps = [ft_metrics['performance/avg_fps']]

x_fps = np.arange(len(fps_labels))

rects1 = ax1.bar(x_fps - width/2, pre_fps, width, label='Pretrained + Text', color='#e74c3c')
rects2 = ax1.bar(x_fps + width/2, ft_fps, width, label='Finetuned + BBox', color='#9b59b6')

ax1.set_ylabel('Frames Per Second')
ax1.set_title('Inference Speed (FPS) - Higher is better')
ax1.set_xticks(x_fps)
ax1.set_xticklabels(fps_labels)
ax1.set_ylim(0, max(max(pre_fps), max(ft_fps)) * 1.2)
ax1.legend()
autolabel_int(rects1, ax1)
autolabel_int(rects2, ax1)

# Latency
lat_labels = ['Latency (ms)']
pre_lat = [pre_metrics['performance/avg_latency_ms']]
ft_lat = [ft_metrics['performance/avg_latency_ms']]

x_lat = np.arange(len(lat_labels))

rects1 = ax2.bar(x_lat - width/2, pre_lat, width, label='Pretrained + Text', color='#e67e22')
rects2 = ax2.bar(x_lat + width/2, ft_lat, width, label='Finetuned + BBox', color='#1abc9c')

ax2.set_ylabel('Milliseconds')
ax2.set_title('Inference Latency (ms) - Lower is better')
ax2.set_xticks(x_lat)
ax2.set_xticklabels(lat_labels)
ax2.set_ylim(0, max(max(pre_lat), max(ft_lat)) * 1.2)
ax2.legend()
autolabel_int(rects1, ax2)
autolabel_int(rects2, ax2)

fig.tight_layout()
plt.savefig(out_dir / "performance_comparison.png", dpi=300)
print(f"Saved plots to {out_dir}")
