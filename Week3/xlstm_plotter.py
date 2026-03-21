import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from dataset import idx2char

def plot_xlstm_visualizations(image_tensor, pseudo_attn_maps, surprise_map, generated_tokens, save_path_prefix):
    """
    Plots the Pseudo-Attention Maps and Surprise Maps.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img = image_tensor.squeeze(0).cpu()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    img_np = img.permute(1, 2, 0).numpy()
    
    H, W = img_np.shape[:2]
    words = [idx2char.get(t.item(), '<UNK>') for t in generated_tokens]
    num_tokens = len(words)
    
    # 1. Plot Surprise Map
    fig, ax = plt.subplots(figsize=(6, 6))
    sal = surprise_map.squeeze().numpy()
    if sal.max() > 0: sal = sal / sal.max()
    sal_resized = cv2.resize(sal, (W, H), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(np.uint8(255 * sal_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[:, :, ::-1] 
    overlay = 0.5 * img_np + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    ax.imshow(overlay)
    ax.set_title("Surprise Map (Delta C_t/h_t) during visual scanning", fontsize=14)
    ax.axis('off')
    surprise_path = save_path_prefix + "_surprise_map.png"
    plt.tight_layout()
    os.makedirs(os.path.dirname(surprise_path), exist_ok=True)
    plt.savefig(surprise_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {surprise_path}")

    # 2. Plot Pseudo-Attention Maps (Cosine Sim)
    cols = 5
    rows = int(np.ceil((num_tokens + 1) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    for i, (word, p_map) in enumerate(zip(words, pseudo_attn_maps)):
        ax = axes[i + 1]
        sal = p_map.squeeze().numpy()
        sal = np.maximum(sal, 0) # ReLU for cosine similarity to ignore negative correlations
        if sal.max() > 0: sal = sal / sal.max()
        sal_resized = cv2.resize(sal, (W, H), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.applyColorMap(np.uint8(255 * sal_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[:, :, ::-1] 
        overlay = 0.5 * img_np + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)
        ax.imshow(overlay)
        ax.set_title(word, fontsize=14)
        ax.axis('off')
        
    for i in range(num_tokens + 1, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    attn_path = save_path_prefix + "_pseudo_attn.png"
    plt.savefig(attn_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {attn_path}")
