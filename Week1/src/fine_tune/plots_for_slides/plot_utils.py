import os
import glob
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import copy

def load_data(results_dir, prefix=''):
    """
    Loads best_metrics.json and config.yaml from each run folder.
    Returns a list of dictionaries with model name and metrics.
    """
    data = []
    
    # Iterate through folders
    search_path = Path(results_dir)
    if not search_path.exists():
        print(f"Error: Directory {results_dir} does not exist.")
        return data
        
    for run_dir in search_path.iterdir():
        if not run_dir.is_dir():
            continue
            
        # Filter by prefix if provided
        if prefix and not run_dir.name.startswith(prefix):
            continue
            
        metrics_file = run_dir / 'best_metrics.json'
        config_file = run_dir / 'config.yaml'
        
        if not metrics_file.exists() or not config_file.exists():
            print(f"Skipping {run_dir.name}: missing best_metrics.json or config.yaml")
            continue
            
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        experiment_name = config.get('experiment_name', run_dir.name)
        # Clean experiment name to act as Model label (e.g. faster_rcnn_Freeze_L1_LR_0.0001 -> Freeze L1 LR 0.0001)
        clean_name = experiment_name.replace("faster_rcnn_", "").replace("rt_detr_", "").replace("rtdetr_", "").replace("detr_", "").replace("yolo_", "").replace("_", " ")
    
        
        data.append({
            'model': clean_name,
            'metrics': metrics,
            'original_dir': run_dir.name
        })
        
    # Sort data by model name
    data.sort(key=lambda x: x['model'])
    return data

def rename_layer_labels(data):
    """
    Renames the layer labels of Faster R-CNN in the data to match the order of DETR and YOLO finetuning strategies.
    """
    new_data = copy.deepcopy(data)
    
    # Define the exact swap rules
    mapping = {
        'L1': 'S3',
        'L2': 'S2',
        'L3': 'S1',
        'L4': 'S0'
    }
    
    pattern = re.compile(r'\b(L1|L2|L3|L4)\b')
    
    def replace_match(match):
        return mapping[match.group(1)]

    # 1. Apply the renaming
    for item in new_data:
        if 'model' in item:
            item['model'] = pattern.sub(replace_match, item['model'])
        
        if 'original_dir' in item:
            item['original_dir'] = pattern.sub(replace_match, item['original_dir'])
            
    # 2. Sort the data
    def sort_key(item):
        model_name = item.get('model', '')
        layer_match = re.search(r'\bL\d\b', model_name)
        layer = layer_match.group(0) if layer_match else ''
        
        return (layer, model_name)

    new_data.sort(key=sort_key)
            
    return new_data

def plot_overall_bar(data, mode="all"):
    models = [item['model'] for item in data]
    
    # Split the x-axis labels into two lines by replacing " LR " with a newline
    formatted_models = [model.replace(" LR ", "\nLR ") for model in models]
    
    # overall/AP and overall/AR_max100
    overall_ap = [item['metrics'].get('overall/AP', 0) * 100 for item in data]
    overall_ar = [item['metrics'].get('overall/AR_max100', item['metrics'].get('overall/AR_max10', 0)) * 100 for item in data]
    
    x = np.arange(len(models))
    
    if mode == "final" or len(models) <= 2:
        width = 0.25
        fig_width = 8
    else:
        width = 0.42 
        fig_width = max(10, len(models) * 1.5) 

    color_ap = "#1f4e79"
    color_ar = "#9fd3c7"

    plt.figure(figsize=(fig_width, 6))

    plt.title("mAP and mAR Comparison", fontsize=20, fontweight="bold", pad=35)

    bars1 = plt.bar(x - width/2, overall_ap, width, label="mAP", color=color_ap)
    bars2 = plt.bar(x + width/2, overall_ar, width, label="mAR", color=color_ar)

    plt.xticks(x, formatted_models, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=14)
    plt.ylim(0, 100)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2,
                         height + 1.5,
                         f"{height:.1f}",
                         ha="center", 
                         fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=14)
    
    plt.show()


def plot_per_class_bar(data, mode="all"):
    plt.style.use('seaborn-v0_8-darkgrid')
    
    models = [item['model'] for item in data]
    
    car_ap = [item['metrics'].get('car/AP', 0) * 100 for item in data]
    ped_ap = [item['metrics'].get('person/AP', 0) * 100 for item in data]
    
    car_ar = [item['metrics'].get('car/AR_max100', 0) * 100 for item in data]
    ped_ar = [item['metrics'].get('person/AR_max100', 0) * 100 for item in data]

    x = np.arange(len(models))

    if mode == "final" or len(models) <= 2:
        width = 0.25
        fig_width = 10
        title_fs = 18
        tick_fs = 14
        label_fs = 14
        text_fs = 12
        legend_fs = 14
    else:
        width = 0.35
        fig_width = max(14, len(models) * 2.5)
        title_fs = 14
        tick_fs = 10
        label_fs = 12
        text_fs = 9
        legend_fs = 13

    color_car = "#1f4e79"
    color_ped = "#9fd3c7"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 6))

    bars1 = ax1.bar(x - width/2, car_ap, width, label='Car', color=color_car)
    bars2 = ax1.bar(x + width/2, ped_ap, width, label='Pedestrian', color=color_ped)

    ax1.set_title('Average Precision (AP) by Class', fontsize=title_fs, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=tick_fs, rotation=45, ha='right')
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Percentage (%)', fontsize=label_fs)
    ax1.tick_params(axis='y', labelsize=tick_fs)

    bars3 = ax2.bar(x - width/2, car_ar, width, label='Car', color=color_car)
    bars4 = ax2.bar(x + width/2, ped_ar, width, label='Pedestrian', color=color_ped)

    ax2.set_title('Average Recall (AR) by Class', fontsize=title_fs, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=tick_fs, rotation=45, ha='right')
    ax2.set_ylim(0, 100)
    ax2.set_yticklabels([])

    def autolabel(bars, ax):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=text_fs, color='black')

    autolabel(bars1, ax1)
    autolabel(bars2, ax1)
    autolabel(bars3, ax2)
    autolabel(bars4, ax2)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=legend_fs)
    
    plt.show()

def plot_overall_heatmap(data, mode="all"):
    models = [item['model'] for item in data]
    
    # overall/AP_small, overall/AP_medium, overall/AP_large, overall/AR_small, overall/AR_medium, overall/AR_large
    matrix = []
    for item in data:
        row = [
            item['metrics'].get('overall/AP_small', 0) * 100,
            item['metrics'].get('overall/AP_medium', 0) * 100,
            item['metrics'].get('overall/AP_large', 0) * 100,
            item['metrics'].get('overall/AR_small', 0) * 100,
            item['metrics'].get('overall/AR_medium', 0) * 100,
            item['metrics'].get('overall/AR_large', 0) * 100
        ]
        matrix.append(row)
        
    data_array = np.array(matrix)
    
    if mode == "final" or len(models) <= 2:
        is_vertical = True
        data_array = data_array.T
        fig_width = max(4, len(models) * 1.5)
        fig_height = 6
        title_fs = 18
        tick_fs = 14
        text_fs = 14
        cbar_fs = 14
    else:
        is_vertical = False
        fig_width = 10
        fig_height = max(4.5, len(models) * 0.5 + 2)
        title_fs = 14
        tick_fs = 10
        text_fs = 10
        cbar_fs = 10

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(data_array, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_title("mAP and mAR by Object Size", fontsize=title_fs, fontweight="bold", pad=15)

    if is_vertical:
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(6))
        ax.set_xticklabels(models, fontsize=tick_fs, rotation=45, ha='right')
        ax.set_yticklabels([
            "AP_S (Small)", "AP_M (Medium)", "AP_L (Large)",
            "AR_S (Small)", "AR_M (Medium)", "AR_L (Large)"
        ], fontsize=tick_fs)
    else:
        ax.set_xticks(np.arange(6))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels([
            "AP_S\n(Small)", "AP_M\n(Medium)", "AP_L\n(Large)",
            "AR_S\n(Small)", "AR_M\n(Medium)", "AR_L\n(Large)"
        ], fontsize=tick_fs)
        ax.set_yticklabels(models, fontsize=tick_fs)

    ax.grid(False) 

    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            val = data_array[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black", fontweight="bold", fontsize=text_fs)

    for spine in ax.spines.values():
        spine.set_visible(False)
        
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Percentage (%)", rotation=-90, va="bottom", labelpad=15, fontsize=cbar_fs)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=cbar_fs)

    plt.tight_layout()
    plt.show()

def plot_per_class_heatmap(data, mode="all"):
    models = [item['model'] for item in data]
    metrics_labels = ["AP_small", "AP_medium", "AP_large", "AR_small", "AR_medium", "AR_large"]
    
    matrix_car = []
    matrix_ped = []
    
    for item in data:
        car_row = [
            item['metrics'].get('car/AP_small', 0) * 100,
            item['metrics'].get('car/AP_medium', 0) * 100,
            item['metrics'].get('car/AP_large', 0) * 100,
            item['metrics'].get('car/AR_small', 0) * 100,
            item['metrics'].get('car/AR_medium', 0) * 100,
            item['metrics'].get('car/AR_large', 0) * 100
        ]
        matrix_car.append(car_row)
        
        ped_row = [
            item['metrics'].get('person/AP_small', 0) * 100,
            item['metrics'].get('person/AP_medium', 0) * 100,
            item['metrics'].get('person/AP_large', 0) * 100,
            item['metrics'].get('person/AR_small', 0) * 100,
            item['metrics'].get('person/AR_medium', 0) * 100,
            item['metrics'].get('person/AR_large', 0) * 100
        ]
        matrix_ped.append(ped_row)
        
    data_car = np.array(matrix_car)
    data_ped = np.array(matrix_ped)

    if mode == "final" or len(models) <= 2:
        fig_height = 3.5
        title_fs = 18
        suptitle_fs = 22
        tick_fs = 14
        ytick_fs = 14
        text_fs = 14
        cbar_fs = 14
    else:
        fig_height = max(5, len(models) * 0.5 + 2)
        title_fs = 15
        suptitle_fs = 18
        tick_fs = 12
        ytick_fs = 11
        text_fs = 11
        cbar_fs = 12

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_height), constrained_layout=True)

    im1 = ax1.imshow(data_car, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    im2 = ax2.imshow(data_ped, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax1.set_xticks(np.arange(len(metrics_labels)))
    ax1.set_yticks(np.arange(len(models)))
    ax1.set_xticklabels(metrics_labels, fontsize=tick_fs)
    ax1.set_yticklabels(models, fontsize=ytick_fs, fontweight="bold")
    ax1.set_title("Cars: Performance by Object Size", fontsize=title_fs, fontweight="bold", pad=15)

    ax2.set_xticks(np.arange(len(metrics_labels)))
    ax2.set_yticks(np.arange(len(models)))
    ax2.set_xticklabels(metrics_labels, fontsize=tick_fs)
    ax2.set_yticklabels(models, fontsize=ytick_fs, fontweight="bold") 
    ax2.set_title("Pedestrians: Performance by Object Size", fontsize=title_fs, fontweight="bold", pad=15)

    for ax, mat_data in zip([ax1, ax2], [data_car, data_ped]):
        ax.grid(False) # Quitar líneas blancas
        for i in range(mat_data.shape[0]):
            for j in range(mat_data.shape[1]):
                val = mat_data[i, j]
                if val > 0:
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black", fontweight="bold", fontsize=text_fs)
        
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.015, pad=0.03)
    cbar.ax.set_ylabel("Percentage (%)", rotation=-90, va="bottom", labelpad=15, fontsize=cbar_fs)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=cbar_fs)

    fig.suptitle("Class-Level Metrics Comparison", fontsize=suptitle_fs, fontweight="bold")
    
    plt.show()
