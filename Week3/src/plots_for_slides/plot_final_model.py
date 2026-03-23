import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model_path = "/ghome/group01/C5/benet/C5-Team1/Week3/results/full_models/clip-vit-b32_lstm_d1024_l3_word_attn_adaptive_256_jvy828i7"
json_file = os.path.join(model_path, 'captions_history.json')

with open(json_file, 'r') as f:
    data = json.load(f)

last_epoch = list(data.keys())[-1]
samples = data[last_epoch]

examples = {
    "Good": ["VizWiz_val_00000102.jpg", "VizWiz_val_00000000.jpg", "VizWiz_val_00000404.jpg"],
    "Bad": ["VizWiz_val_00000354.jpg", "VizWiz_val_00000304.jpg", "VizWiz_val_00000254.jpg"]
}

output_dir = "/ghome/group01/C5/vali/C5-Team1/Week3/results/qualitative_final_model"
os.makedirs(output_dir, exist_ok=True)

# create mapping
sample_dict = {s["image_name"]: s for s in samples}

for category, img_list in examples.items():
    for img_name in img_list:
        s = sample_dict[img_name]
        
        fig = plt.figure(figsize=(24, 10))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2])
        
        # Original Image
        ax0 = fig.add_subplot(gs[0])
        img_path = os.path.join(model_path, "visual_samples", img_name)
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax0.imshow(img)
        ax0.axis('off')
        ax0.set_title(f"Original Image ({category} Example)", fontsize=16, fontweight='bold')
        
        # Text
        ax1 = fig.add_subplot(gs[1])
        ax1.axis('off')
        
        y_pos = 0.95
        ax1.text(0.05, y_pos, "Final Model Prediction:", fontsize=18, fontweight='bold', va='top')
        y_pos -= 0.08
        ax1.text(0.1, y_pos, s['prediction'], fontsize=16, va='top', wrap=True, color='blue' if category=='Good' else 'red')
        
        # Better dynamic y_pos update
        prediction_length = len(s['prediction'])
        y_pos -= 0.1 * (1 + prediction_length // 40)
        
        ax1.text(0.05, y_pos, "Ground Truths:", fontsize=18, fontweight='bold', va='top')
        y_pos -= 0.08
        for i, ref in enumerate(s['references'][:5]):
            ax1.text(0.1, y_pos, f"{i+1}. {ref}", fontsize=14, va='top', wrap=True)
            y_pos -= 0.06 * (1 + len(ref)//50)
            
        # Attention Grid
        ax2 = fig.add_subplot(gs[2])
        attention_dir = os.path.join(model_path, f"attention_outputs_epoch_{int(last_epoch):02d}", img_name.replace(".jpg", "_outputs"))
        grid_file = os.path.join(attention_dir, "adaptive_colored_grid.png")
        if os.path.exists(grid_file):
            grid_img = mpimg.imread(grid_file)
            ax2.imshow(grid_img)
            ax2.set_title("Visual Attention Map (per token)", fontsize=16, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{category}_{img_name.replace('.jpg', '.png')}")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()

print(f"Saved {len(examples['Good']) + len(examples['Bad'])} plots to {output_dir}")
