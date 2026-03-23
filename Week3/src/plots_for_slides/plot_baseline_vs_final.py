import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Paths
baseline_path = "/ghome/group01/C5/benet/C5-Team1/Week3/results/baseline/full/resnet18_gru_d512_l1_char_8neq9rr1"
final_path = "/ghome/group01/C5/vali/C5-Team1/Week3/results/full_sweep/clip-vit-b32_lstm_d1024_l3_word_attn_adaptive_256_jvy828i7"

baseline_json = os.path.join(baseline_path, 'captions_history.json')
final_json = os.path.join(final_path, 'captions_history.json')

with open(baseline_json, 'r') as f:
    baseline_data = json.load(f)
with open(final_json, 'r') as f:
    final_data = json.load(f)

# Request specified Epoch 7 for the final model
final_epoch = "7"
# We will use the last epoch (10) for the baseline
baseline_epoch = str(sorted([int(e) for e in baseline_data.keys()])[-1])

output_dir = "/ghome/group01/C5/vali/C5-Team1/Week3/results/qualitative_baseline_vs_final"
os.makedirs(output_dir, exist_ok=True)

# Build a dictionary of images to their predictions
images_dict = {}

for s in baseline_data[baseline_epoch]:
    img_name = s['image_name']
    images_dict[img_name] = {
        'references': s['references'],
        'baseline_pred': s['prediction']
    }

for s in final_data[final_epoch]:
    img_name = s['image_name']
    if img_name in images_dict:
        images_dict[img_name]['final_pred'] = s['prediction']

for img_name, data in images_dict.items():
    if 'final_pred' not in data:
        continue # Skip if missing for some reason
        
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1.5]})
    
    # Original Image
    # Try finding the image in the final model's visual_samples folder
    img_path = os.path.join(final_path, "visual_samples", img_name)
    if not os.path.exists(img_path):
        img_path = os.path.join(baseline_path, "visual_samples", img_name)
        
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(f"Image: {img_name}", fontsize=16, fontweight='bold')
    
    # Text
    axes[1].axis('off')
    
    y_pos = 0.95
    axes[1].text(0.05, y_pos, "Ground Truths:", fontsize=18, fontweight='bold', va='top')
    y_pos -= 0.08
    for i, ref in enumerate(data['references'][:5]):
        axes[1].text(0.1, y_pos, f"{i+1}. {ref}", fontsize=14, va='top', wrap=True)
        y_pos -= 0.06 * (1 + len(ref)//60)
        
    y_pos -= 0.04
    axes[1].text(0.05, y_pos, f"Final Baseline Prediction (Epoch {baseline_epoch}):", fontsize=18, fontweight='bold', va='top', color='red')
    y_pos -= 0.06
    axes[1].text(0.1, y_pos, data['baseline_pred'], fontsize=16, fontweight='normal', va='top', wrap=True, color='red')
    
    y_pos -= 0.1 * (1 + len(data['baseline_pred'])//60)
    
    axes[1].text(0.05, y_pos, f"Final Model Prediction (Epoch {final_epoch}):", fontsize=18, fontweight='bold', va='top', color='blue')
    y_pos -= 0.06
    axes[1].text(0.1, y_pos, data['final_pred'], fontsize=16, fontweight='bold', va='top', wrap=True, color='blue')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"Comparison_{img_name.replace('.jpg', '.png')}")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

print(f"Saved {len(images_dict)} comparison plots to {output_dir}")
