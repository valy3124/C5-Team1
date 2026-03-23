import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Model configuration
models = [
    {"name": "Baseline", "epoch": "9", "path": "/ghome/group01/C5/benet/C5-Team1/Week3/results/best_each_step/resnet18_lr0.0005_bs128_cmiqlnrw"},
    {"name": "Encoder", "epoch": "6", "path": "/ghome/group01/C5/benet/C5-Team1/Week3/results/best_each_step/clip-vit-b32_lr0.0005_bs128_fo4ry8j4"},
    {"name": "Decoder", "epoch": "9", "path": "/ghome/group01/C5/benet/C5-Team1/Week3/results/best_each_step/clip-vit-b32_lstm_d1024_l3_lr0.0005_bs128_xyl49ynt"},
    {"name": "Text-level", "epoch": "9", "path": "/ghome/group01/C5/benet/C5-Team1/Week3/results/best_each_step/clip-vit-b32_lstm_d1024_l3_word_txvpvjk8"},
    {"name": "Attention", "epoch": "7", "path": "/ghome/group01/C5/benet/C5-Team1/Week3/results/best_each_step/clip-vit-b32_lstm_d1024_l3_word_attn_adaptive_256_u4a0z6p8"},
    {"name": "Sweep", "epoch": "7", "path": "/ghome/group01/C5/benet/C5-Team1/Week3/results/best_each_step/best_sweep"}
]

# Gathering the data
image_data = {}
for m in models:
    json_file = os.path.join(m['path'], 'captions_history.json')
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
            epoch_data = data.get(m['epoch'], [])
            for item in epoch_data:
                img_name = item['image_name']
                if img_name not in image_data:
                    image_data[img_name] = {
                        'references': item['references'],
                        'predictions': {},
                        'img_path': os.path.join(m['path'], 'visual_samples', img_name)
                    }
                image_data[img_name]['predictions'][m['name']] = item['prediction']

# Output directory for the plots
output_dir = "/ghome/group01/C5/vali/C5-Team1/Week3/results/qualitative_comparisons"
os.makedirs(output_dir, exist_ok=True)

for img_name, data in image_data.items():
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 2]})
    
    # Plot image
    if os.path.exists(data['img_path']):
        img = mpimg.imread(data['img_path'])
        axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(img_name, fontsize=14, fontweight='bold')
    
    # Plot text
    axes[1].axis('off')
    
    text_content = ""
    
    # Ground Truths
    text_content += "\\textbf{Ground Truths:}\n"
    for i, ref in enumerate(data['references']):
        text_content += f"  {i+1}. {ref}\n"
        
    text_content += "\n\\textbf{Model Predictions:}\n"
    for m in models:
        pred = data['predictions'].get(m['name'], "N/A")
        text_content += f"  \\textbf{{{m['name']} (Epoch {m['epoch']})}}: {pred}\n\n"
        
    # Using markdown parsing in matplotlib isn't out-of-the-box easy for bold text without LaTeX, 
    # so we will just use formatting or pure text if LaTeX is not enabled.
    # Let's do raw text with distinct styling per section.
    
    # Clearer separation without raw LaTeX
    y_pos = 0.95
    axes[1].text(0.05, y_pos, "Ground Truths:", fontsize=16, fontweight='bold', va='top')
    y_pos -= 0.05
    for i, ref in enumerate(data['references']):
        axes[1].text(0.08, y_pos, f"{i+1}. {ref}", fontsize=14, va='top', wrap=True)
        # Approximate line height
        y_pos -= 0.04 * (1 + len(ref)//80)
        
    y_pos -= 0.05
    axes[1].text(0.05, y_pos, "Model Predictions:", fontsize=16, fontweight='bold', va='top')
    y_pos -= 0.05
    for m in models:
        pred = data['predictions'].get(m['name'], "N/A")
        # Bold model name, normal prediction
        axes[1].text(0.08, y_pos, f"{m['name']} (Ep {m['epoch']}):", fontsize=14, fontweight='bold', va='top')
        axes[1].text(0.35, y_pos, f"{pred}", fontsize=14, va='top', wrap=True)
        y_pos -= 0.04 * (1 + len(pred)//60)
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, img_name.replace('.jpg', '.png'))
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

print(f"Saved {len(image_data)} qualitative plots to {output_dir}")
