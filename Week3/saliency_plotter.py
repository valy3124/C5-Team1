import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image
from dataset import idx2char

def plot_saliency_maps(image_tensor, saliency_maps, generated_tokens, save_path):
    """
    Plots the Saliency Maps computed from Input Gradients overlaying the original image.
    Args:
        image_tensor: (1, 3, H, W) normalized image tensor that was passed to the model
        saliency_maps: list of (1, 1, H, W) gradient tensors for each token
        generated_tokens: list of token IDs
        save_path: string path to save the resulting plot
    """
    # Denormalize image for plotting (assuming ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img = image_tensor.squeeze(0).cpu()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    img_np = img.permute(1, 2, 0).numpy()
    
    H, W = img_np.shape[:2]
    
    # Text decode
    words = [idx2char.get(t.item(), '<UNK>') for t in generated_tokens]
    
    num_tokens = len(words)
    cols = 5
    rows = int(np.ceil((num_tokens + 1) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()
    
    # Plot original image in the first subplot
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    # Plot each token's saliency map
    for i, (word, saliency) in enumerate(zip(words, saliency_maps)):
        ax = axes[i + 1]
        
        # Format saliency
        sal = saliency.squeeze().numpy()
        
        # Normalize saliency to 0-1 for visualization
        if sal.max() > 0:
            sal = sal / sal.max()
        
        # Resize saliency to match image dimensions
        sal_resized = cv2.resize(sal, (W, H), interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * sal_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # Overlay heatmap on original image (BGR to RGB conversion for cv2 map)
        heatmap = heatmap[:, :, ::-1] 
        overlay = 0.5 * img_np + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)
        
        ax.imshow(overlay)
        ax.set_title(word, fontsize=14)
        ax.axis('off')
        
    # Turn off axis for unused subplots
    for i in range(num_tokens + 1, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
if __name__ == "__main__":
    print("Saliency plotter module loaded.")
