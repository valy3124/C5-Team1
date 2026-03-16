import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pycocotools.mask as rletools
from pathlib import Path

# Create output dir
out_dir = Path("/ghome/group01/C5/vali/C5-Team1/Week2/results_semantic/plots")
out_dir.mkdir(exist_ok=True, parents=True)

import sys
from PIL import Image

# Import the KITTI-MOTS dataset loader to get real images & masks
ROOT_DIR = Path("/ghome/group01/C5/vali/C5-Team1/Week2")
sys.path.append(str(ROOT_DIR))
from src.datasets import KITTIMOTS

def get_real_scene():
    dataset = KITTIMOTS(
        root="~/mcv/datasets/C5/KITTI-MOTS/",
        split="validation", 
        ann_source="txt",
        compute_boxes=True
    )
    
    # Let's try an index with closer overlapping objects (e.g. index 75 or 150)
    demo_index = 150
    
    image_pil, anns, meta = dataset[demo_index]
    img_np = np.array(image_pil)
    
    inst_masks = []
    class_ids = []
    
    # We will compute the absolute min/max coordinates of ALL masks 
    # so we can crop the empty background away
    min_x, min_y = 9999, 9999
    max_x, max_y = 0, 0
    
    for ann in anns:
        mask = rletools.decode(ann.mask_rle).astype(np.uint8)
        
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            min_x = min(min_x, xs.min())
            min_y = min(min_y, ys.min())
            max_x = max(max_x, xs.max())
            max_y = max(max_y, ys.max())
            
        inst_masks.append(mask)
        target_cid = dataset.LABELS_MAPPING.get(ann.class_id, ann.class_id)
        class_ids.append(target_cid)
        
    # If we found objects, crop around them with some generous padding
    if max_x > min_x:
        pad_x, pad_y = 150, 100
        H, W = img_np.shape[:2]
        
        c_min_x = max(0, min_x - pad_x)
        c_min_y = max(0, min_y - pad_y)
        c_max_x = min(W, max_x + pad_x)
        c_max_y = min(H, max_y + pad_y)
        
        # Apply crop
        img_np = img_np[c_min_y:c_max_y, c_min_x:c_max_x]
        inst_masks = [m[c_min_y:c_max_y, c_min_x:c_max_x] for m in inst_masks]
        
    return img_np, inst_masks, class_ids

img, inst_masks, class_ids = get_real_scene()

# Make the Figure less outrageously wide since we cropped
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(wspace=0.1)

# Panel 1: Original Image Representation
axs[0].imshow(img)
axs[0].set_title("1. Original Image (RGB)", fontsize=16, pad=20)
axs[0].axis('off')

# Panel 2: Instance Segmentation (KITTI-MOTS Default)
axs[1].imshow(img)

# Dynamic colours based on class
import matplotlib.colors as mcolors
person_color = '#e74c3c' # Red
car_color = '#3498db'    # Blue
alphas = 0.6

# Draw instances
for i, (mask, cid) in enumerate(zip(inst_masks, class_ids)):
    color = person_color if cid == 1 else car_color
    cmap = mcolors.ListedColormap(['none', color])
    axs[1].imshow(mask, cmap=cmap, alpha=alphas)
    
    # Draw dashed bounding boxes for impact
    ys, xs = np.where(mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        rect = patches.Rectangle((xs.min(), ys.min()), xs.max()-xs.min(), ys.max()-ys.min(), 
                                 linewidth=1.5, edgecolor=color, facecolor='none', linestyle='--')
        axs[1].add_patch(rect)

axs[1].set_title("2. Instance Annotations (Source Dataset)", fontsize=16, pad=20)
axs[1].axis('off')

# Add a small legend for the instance panel
legend_elements = [
    patches.Patch(facecolor=person_color, alpha=0.6, label='Person Instance'),
    patches.Patch(facecolor=car_color, alpha=0.6, label='Car Instance')
]
axs[1].legend(handles=legend_elements, loc='upper right', prop={'size': 7}, handlelength=1.0)

# Panel 3: Semantic Segmentation (Our Target)
# The preprocessing step: _collapse_masks_to_class_union
semantic_map = np.zeros(img.shape[:2], dtype=np.int32)
label_colors = {0: [0, 0, 0], 1: [220/255, 50/255, 50/255], 3: [50/255, 100/255, 220/255]} # BG, Person, Car

for m, cid in zip(inst_masks, class_ids):
    semantic_map[m == 1] = cid

# To make the semantic map look nice and opaque against a dark background, 
# we'll plot the raw semantic mask without the image behind it
semantic_rgb = np.zeros((img.shape[0], img.shape[1], 3))
for cid, col in label_colors.items():
    semantic_rgb[semantic_map == cid] = col

axs[2].imshow(semantic_rgb)
axs[2].set_title("3. Semantic Preprocessing (Target/Input)", fontsize=16, pad=20)

axs[2].axis('off')

# Add text explaining the union logic
axs[2].text(img.shape[1]//2, img.shape[0] - 50, 
            "Overlapping instances are merged via Logical-OR\ninto flat Class Maps for Semantic finetuning.", 
            color='white', weight='bold', ha='center', va='center', 
            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.savefig(out_dir / "semantic_preprocessing_flow.png", dpi=300, bbox_inches='tight')
print(f"Saved real-image visualization to {out_dir / 'semantic_preprocessing_flow.png'}")
