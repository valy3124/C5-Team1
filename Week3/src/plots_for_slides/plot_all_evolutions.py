import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Correct path using vali
model_path = "/ghome/group01/C5/vali/C5-Team1/Week3/results/full_sweep/clip-vit-b32_lstm_d1024_l3_word_attn_adaptive_256_jvy828i7"
json_file = os.path.join(model_path, 'captions_history.json')

with open(json_file, 'r') as f:
    data = json.load(f)

# Sort epochs
epochs = sorted([int(e) for e in data.keys()])

# Get all unique images from the last epoch
target_images = [s['image_name'] for s in data[str(epochs[-1])]]

output_dir = "/ghome/group01/C5/vali/C5-Team1/Week3/results/qualitative_epoch_evolution_all"
os.makedirs(output_dir, exist_ok=True)

for img_name in target_images:
    
    # Extract references
    references = []
    for s in data[str(epochs[-1])]:
        if s['image_name'] == img_name:
            references = s['references'][:4] # Take up to 4 ground truths
            break
            
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [1, 1.5]})
    
    # Original Image
    img_path = os.path.join(model_path, "visual_samples", img_name)
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(f"Image: {img_name}", fontsize=16, fontweight='bold')
    
    # Text
    axes[1].axis('off')
    
    y_pos = 0.95
    axes[1].text(0.05, y_pos, "Ground Truths:", fontsize=18, fontweight='bold', va='top')
    y_pos -= 0.05
    for i, ref in enumerate(references):
        axes[1].text(0.1, y_pos, f"{i+1}. {ref}", fontsize=14, va='top', wrap=True)
        y_pos -= 0.04 * (1 + len(ref)//60)
        
    y_pos -= 0.04
    axes[1].text(0.05, y_pos, "Predicted Caption per Epoch:", fontsize=18, fontweight='bold', va='top')
    y_pos -= 0.05
    
    for epoch in epochs:
        # Find prediction for this image in this epoch
        pred = "N/A"
        for s in data[str(epoch)]:
            if s['image_name'] == img_name:
                pred = s['prediction']
                break
                
        color = 'blue' if epoch == epochs[-1] else 'black'
        weight = 'bold' if epoch == epochs[-1] else 'normal'
        
        axes[1].text(0.08, y_pos, f"Epoch {epoch:02d}:", fontsize=14, fontweight='bold', va='top')
        axes[1].text(0.25, y_pos, pred, fontsize=14, fontweight=weight, color=color, va='top', wrap=True)
        
        y_pos -= 0.04 * (1 + len(pred)//60)
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"Evolution_{img_name.replace('.jpg', '.png')}")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

print(f"Saved {len(target_images)} evolution plots to {output_dir}")
