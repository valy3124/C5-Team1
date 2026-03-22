import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Global configuration similar to plot_utils.py
TITLE_FS = 20
LABEL_FS = 14
TICK_FS = 12
BAR_FS = 11

matplotlib.rcParams.update({
    "font.size": LABEL_FS,
    "axes.titlesize": TITLE_FS,
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
})

COLORS = {
    "run1": "#8ecae6",
    "run2": "#023047",
    "metrics": ["#1f4e79", "#2d9e6b", "#e86c1f", "#7b2d8b"]
}

METRIC_KEYS = ["ROUGE-L", "METEOR", "BLEU-1", "BLEU-2"]

RUNS = [
    {
        "name": "Replicated Images (All Captions)",
        "path": "/ghome/group01/C5/benet/C5-Team1/Week3/results/baseline/resnet18_lr0.0005_bs128_cmiqlnrw/best_metrics.json"
    },
    {
        "name": "Random Caption / Image",
        "path": "/ghome/group01/C5/benet/C5-Team1/Week3/results/baseline/resnet18_lr0.001_bs32_zzi9u8ua/best_metrics.json"
    }
]

def load_metrics(runs):
    loaded_runs = []
    for run in runs:
        with open(run["path"], "r") as f:
            data = json.load(f)
            loaded_runs.append({
                "name": run["name"],
                "metrics": {k: data[k] for k in METRIC_KEYS}
            })
    return loaded_runs

def plot_grouped_by_metric(loaded_runs, output_path):
    x = np.arange(len(METRIC_KEYS))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, run in enumerate(loaded_runs):
        vals = [run["metrics"][k] for k in METRIC_KEYS]
        offset = (i - 0.5) * width
        color = COLORS["run1"] if i == 0 else COLORS["run2"]
        bars = ax.bar(x + offset, vals, width, label=run["name"], color=color, edgecolor="white")
        
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.1f}", ha="center", va="bottom", fontsize=BAR_FS)
            
    ax.set_title("Metrics comparison between baseline experiments", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_KEYS)
    ax.set_ylabel("Score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot by metric: {output_path}")

def plot_grouped_by_experiment(loaded_runs, output_path):
    x = np.arange(len(loaded_runs))
    num_metrics = len(METRIC_KEYS)
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, metric in enumerate(METRIC_KEYS):
        vals = [run["metrics"][metric] for run in loaded_runs]
        offset = (i - (num_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=metric, color=COLORS["metrics"][i], edgecolor="white")
        
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.1f}", ha="center", va="bottom", fontsize=BAR_FS)
            
    ax.set_title("Metrics per baseline experiment", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([run["name"] for run in loaded_runs])
    ax.set_ylabel("Score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot by experiment: {output_path}")

if __name__ == "__main__":
    output_dir = "/ghome/group01/C5/benet/C5-Team1/Week3/results/baseline/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    data = load_metrics(RUNS)
    
    plot_grouped_by_metric(data, os.path.join(output_dir, "metrics_comparison.png"))
    plot_grouped_by_experiment(data, os.path.join(output_dir, "experiment_metrics.png"))
