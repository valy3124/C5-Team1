# Global configuration (Optimized for massive slide visibility)
TITLE_FS = 70  
LABEL_FS = 50  
TICK_FS = 45   
BAR_FS = 35    
LEGEND_FS = 45 

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

matplotlib.rcParams.update({
    "font.size": LABEL_FS,
    "axes.titlesize": TITLE_FS,
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
})

# Color palettes
RUN_COLORS = ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500", "#e63946", "#a8dadc", "#457b9d", "#1d3557"]
METRIC_COLORS = ["#1f4e79", "#2d9e6b", "#e86c1f", "#7b2d8b", "#9b59b6"]

METRIC_KEYS = ["ROUGE-L", "METEOR", "BLEU-1", "BLEU-2", "CIDEr"]

# Refined Attention Colors & Order
ATTN_COLORS = {
    "No Attention": "#8ecae6",
    "Soft": "#219ebc",
    "Adaptive": "#023047",
    "Early Fusion": "#ffb703"
}
ATTN_ORDER = {
    "No Attention": 0,
    "Soft": 1,
    "Adaptive": 2,
    "Early Fusion": 3
}

def load_metrics(results_dir, manual_labels=None, mode="bar", filter_str=None, label_key=None):
    loaded_runs = []
    results_path = Path(results_dir)
    
    # Discovery: find all subdirectories containing best_metrics.json
    subdirs = sorted([d for d in results_path.iterdir() if d.is_dir() and (d / "best_metrics.json").exists()])
    
    for subdir in subdirs:
        if subdir.name == "old":
            continue
            
        if filter_str:
            parts = subdir.name.split("_")
            if not any(filter_str.lower() == p.lower() for p in parts):
                continue
            
        metrics_file = subdir / "best_metrics.json"
        with open(metrics_file, "r") as f:
            data = json.load(f)
            
        run_info = {
            "folder_name": subdir.name,
            "metrics": {k: data.get(k, 0.0) for k in METRIC_KEYS}
        }
        
        # Metadata parsing
        parts = subdir.name.split("_")
        # Pattern: IMAGE-ENCODER_DECODER-TYPE_dDIM_lLAYERS_TEXT-LEVEL_...
        if len(parts) >= 2:
            try:
                run_info["decoder_type"] = parts[1].upper() if len(parts) > 1 else parts[0].upper()
                if len(parts) >= 5:
                    run_info["decoder_dim"] = parts[2][1:] if parts[2].startswith('d') else parts[2]
                    run_info["num_layers"] = parts[3][1:] if parts[3].startswith('l') else parts[3]
                    run_info["text_level"] = parts[4]
            except (IndexError, ValueError):
                pass
        
        # Attention parsing
        if "_attn_" in subdir.name:
            try:
                attn_idx = parts.index("attn")
                run_info["attn_type"] = parts[attn_idx+1].replace("-", " ").title()
            except (ValueError, IndexError):
                run_info["attn_type"] = "Attention"
        else:
            run_info["attn_type"] = "No Attention"
            
        # Determine label
        if manual_labels and subdir.name in manual_labels:
            label = manual_labels[subdir.name]
        elif label_key and label_key in run_info:
            label = run_info[label_key]
        else:
            if mode == "bar" and filter_str:
                label = run_info.get("attn_type", "Model")
            elif mode == "single_metric":
                decoder = run_info.get("decoder_type", "Model")
                attn = run_info.get("attn_type", "")
                label = f"{decoder} + {attn}" if attn != "No Attention" else f"{decoder} (No Attention)"
            else:
                label = subdir.name.split("_")[0]
            
        run_info["name"] = label
        loaded_runs.append(run_info)
    
    # Sort by attention order if possible
    if any("attn" in r.get("folder_name", "").lower() for r in loaded_runs):
        loaded_runs.sort(key=lambda r: (r.get("decoder_type", ""), ATTN_ORDER.get(r.get("attn_type"), 99)))
    
    return loaded_runs

def plot_grouped_by_metric(loaded_runs, output_path, title="Metrics Comparison"):
    num_runs = len(loaded_runs)
    num_metrics = len(METRIC_KEYS)
    x = np.arange(num_metrics)
    
    width = 0.9 / (num_runs + 0.1) if num_runs > 1 else 0.5
    fig_width = max(24, num_runs * 8.0)
    fig_height = 18 
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height)) 
    
    for i, run in enumerate(loaded_runs):
        vals = [run["metrics"].get(k, 0.0) for k in METRIC_KEYS]
        offset = (i - (num_runs - 1) / 2) * width
        
        color = ATTN_COLORS.get(run.get("attn_type"), RUN_COLORS[i % len(RUN_COLORS)])
        bars = ax.bar(x + offset, vals, width, label=run["name"], color=color, edgecolor="white", linewidth=0.5)
        
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", 
                        ha="center", va="bottom", fontsize=BAR_FS, fontweight='bold')
            
    ax.set_title(title, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_KEYS)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max([max(list(r["metrics"].values()) + [0]) for r in loaded_runs] + [10]) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot by metric: {output_path}")

def plot_single_metric_comparison(loaded_runs, output_path, metric_key="METEOR", title=None):
    num_runs = len(loaded_runs)
    x = np.arange(num_runs)
    width = 0.6
    fig_width = max(24, num_runs * 6.0)
    fig_height = 18
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    vals = [run["metrics"].get(metric_key, 0.0) for run in loaded_runs]
    colors = [ATTN_COLORS.get(run.get("attn_type"), RUN_COLORS[i % len(RUN_COLORS)]) for i, run in enumerate(loaded_runs)]
    bars = ax.bar(x, vals, width, color=colors, edgecolor="white", linewidth=2.0)
    from matplotlib.lines import Line2D
    present_attn = sorted(list(set([r.get("attn_type") for r in loaded_runs])), key=lambda a: ATTN_ORDER.get(a, 99))
    if len(present_attn) > 1:
        legend_elements = [Line2D([0], [0], color=ATTN_COLORS.get(a), lw=10, label=a) for a in present_attn]
        ax.legend(handles=legend_elements, title="Attention Type", bbox_to_anchor=(1.02, 1), loc='upper left')
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=BAR_FS, fontweight='bold')
    if not title: title = f"{metric_key} Comparison"
    ax.set_title(title, fontweight="bold", pad=30)
    ax.set_xticks(x); ax.set_xticklabels([f"{r.get('decoder_type')} - {r.get('attn_type')}" for r in loaded_runs], rotation=45, ha="right")
    ax.set_ylabel(metric_key); ax.set_ylim(0, max(vals + [10]) * 1.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close(); print(f"Saved single metric plot: {output_path}")

def plot_dev_eval_comparison(loaded_runs, output_dir):
    # Specialized mode for Full Models (Baseline/Final) comparing Dev vs Eval
    # Groups: Baseline (dev, eval), Final (dev, eval)
    models = ["Baseline", "Final Model"]
    sets = ["Dev", "Eval"]
    
    # Organize data
    data_map = {}
    for r in loaded_runs:
        folder = r["folder_name"].lower()
        if "baseline" in folder:
            m = "Baseline"
        elif "final" in folder:
            m = "Final Model"
        else: continue
        
        s = "Eval" if "eval" in folder else "Dev"
        if m not in data_map: data_map[m] = {}
        data_map[m][s] = r["metrics"]
    
    y_limit = 70
    
    for model_name in data_map:
        if len(data_map[model_name]) < 2: continue
        
        dev_scores = [data_map[model_name]["Dev"].get(k, 0.0) for k in METRIC_KEYS]
        eval_scores = [data_map[model_name]["Eval"].get(k, 0.0) for k in METRIC_KEYS]
        
        num_metrics = len(METRIC_KEYS)
        x = np.arange(num_metrics)
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(24, 16))
        rects1 = ax.bar(x - width/2, dev_scores, width, label='Dev', color="#a8dadc", edgecolor="white")
        rects2 = ax.bar(x + width/2, eval_scores, width, label='Eval', color="#457b9d", edgecolor="white")
        
        # Add labels and indicators
        for i in range(num_metrics):
            d, e = dev_scores[i], eval_scores[i]
            # Use rounded values for consistency as requested
            d_rounded = round(d, 1)
            e_rounded = round(e, 1)
            
            # Dev label
            ax.text(x[i] - width/2, d + 0.5, f"{d_rounded:.1f}", ha='center', va='bottom', fontsize=BAR_FS-10)
            # Eval label
            ax.text(x[i] + width/2, e + 0.5, f"{e_rounded:.1f}", ha='center', va='bottom', fontsize=BAR_FS-10)
            
            # +/- Indicator Box (Calculated from rounded values)
            diff = e_rounded - d_rounded
            indicator = f"+{diff:.1f}" if diff >= 0 else f"{diff:.1f}"
            color = "#2ecc71" if diff >= 0 else "#e74c3c"
            
            # Draw a small box above the eval bar
            ax.text(x[i] + width/2, e + 5, indicator, ha='center', va='bottom', 
                    fontsize=BAR_FS-5, fontweight='bold', color='white',
                    bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3'))
            
        ax.set_ylabel('Scores')
        ax.set_title(f'{model_name}: Dev vs Eval Comparison', fontweight='bold', pad=40)
        ax.set_xticks(x)
        ax.set_xticklabels(METRIC_KEYS)
        ax.legend(fontsize=LEGEND_FS)
        ax.set_ylim(0, y_limit)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        plt.tight_layout()
        filename = f"{model_name.lower().replace(' ', '_')}_comparison.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved {model_name} comparison plot: {output_path}")

def plot_decoder_heatmaps(loaded_runs, output_dir):
    dims = sorted(list(set([r["decoder_dim"] for r in loaded_runs if "decoder_dim" in r])))
    types = sorted(list(set([r["decoder_type"] for r in loaded_runs if "decoder_type" in r])))
    layers = sorted(list(set([r["num_layers"] for r in loaded_runs if "num_layers" in r])))
    for dim in dims:
        for metric in METRIC_KEYS:
            matrix = np.zeros((len(layers), len(types)))
            mask = np.ones((len(layers), len(types)), dtype=bool)
            for r in loaded_runs:
                if r.get("decoder_dim") == dim and "num_layers" in r and "decoder_type" in r:
                    try:
                        row = layers.index(r["num_layers"]); col = types.index(r["decoder_type"])
                        matrix[row, col] = r["metrics"].get(metric, 0.0); mask[row, col] = False
                    except ValueError: continue
            if np.all(mask): continue
            fig, ax = plt.subplots(figsize=(18, 14))
            im = ax.imshow(matrix, cmap="RdYlGn", vmin=25, vmax=40)
            cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046, extend='both'); cbar.ax.tick_params(labelsize=TICK_FS)
            ax.set_xticks(np.arange(len(types))); ax.set_yticks(np.arange(len(layers)))
            ax.set_xticklabels(types); ax.set_yticklabels([f"{l} Layers" for l in layers])
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            for i in range(len(layers)):
                for j in range(len(types)):
                    if not mask[i, j]:
                        ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="black", fontweight="bold", fontsize=BAR_FS + 10)
            ax.set_title(f"{metric} - Decoder Dim {dim}", fontweight="bold", pad=30)
            ax.set_xlabel("Decoder Type", labelpad=20); ax.set_ylabel("Complexity", labelpad=20)
            fig.tight_layout(); plt.savefig(os.path.join(output_dir, f"heatmap_{metric.lower().replace('-', '_')}_d{dim}.png"), dpi=150); plt.close()

def find_best_runs_per_type(loaded_runs, metric_key="METEOR"):
    best_runs = {}
    for run in loaded_runs:
        d_type = run.get("decoder_type")
        if not d_type: continue
        score = run["metrics"].get(metric_key, 0.0)
        if d_type not in best_runs or score > best_runs[d_type]["metrics"].get(metric_key, 0.0):
            best_runs[d_type] = run
    return list(best_runs.values())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare metrics across experiments.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing experiment subdirectories.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots.")
    parser.add_argument("--title", type=str, default="Sweep Metrics Comparison", help="Title for the plots.")
    parser.add_argument("--manual_labels", type=str, help="JSON string mapping folder names to labels.")
    parser.add_argument("--mode", type=str, default="bar", choices=["bar", "heatmap", "best_decoders", "single_metric", "dev_eval"], help="Plotting mode.")
    parser.add_argument("--metric", type=str, default="METEOR", help="Metric to use for comparison.")
    parser.add_argument("--filter", type=str, help="Substring filter for folder names.")
    parser.add_argument("--label_key", type=str, help="Metadata key to use for labels.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    labels = json.loads(args.manual_labels) if args.manual_labels else None
    data = load_metrics(args.results_dir, labels, mode=args.mode, filter_str=args.filter, label_key=args.label_key)
    
    if not data:
        print(f"No experiments found in {args.results_dir}")
    else:
        if args.mode == "bar":
            plot_grouped_by_metric(data, os.path.join(args.output_dir, "metrics_comparison.png"), title=args.title)
        elif args.mode == "heatmap":
            plot_decoder_heatmaps(data, args.output_dir)
        elif args.mode == "best_decoders":
            best_runs = find_best_runs_per_type(data, metric_key=args.metric)
            if best_runs:
                plot_grouped_by_metric(best_runs, os.path.join(args.output_dir, "best_decoders_by_metric.png"), title=f"Best Decoders Comparison (Ranked by {args.metric})")
        elif args.mode == "single_metric":
            plot_single_metric_comparison(data, os.path.join(args.output_dir, f"all_{args.metric.lower().replace('-', '_')}_comparison.png"), args.metric, title=args.title)
        elif args.mode == "dev_eval":
            plot_dev_eval_comparison(data, args.output_dir)