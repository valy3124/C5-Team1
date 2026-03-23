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

output_dir = "/ghome/group01/C5/vali/C5-Team1/Week3/results/qualitative_final_model_simple"
os.makedirs(output_dir, exist_ok=True)

# create mapping
sample_dict = {s["image_name"]: s for s in samples}

for category, img_list in examples.items():
    for img_name in img_list:
        if img_name not in sample_dict:
            continue
        s = sample_dict[img_name]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1.5]})
        
        # Original Image
        img_path = os.path.join(model_path, "visual_samples", img_name)
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title(f"Image ({category} Example)", fontsize=16, fontweight='bold')
        
        # Text
        axes[1].axis('off')
        
        y_pos = 0.95
        axes[1].text(0.05, y_pos, "Predicted Caption:", fontsize=18, fontweight='bold', va='top')
        y_pos -= 0.08
        axes[1].text(0.1, y_pos, s['prediction'], fontsize=16, va='top', wrap=True, color='blue' if category=='Good' else 'red')
        
        # Adjust spacing based on text length
        prediction_length = len(s['prediction'])
        y_pos -= 0.1 * (1 + prediction_length // 40)
        
        axes[1].text(0.05, y_pos, "Ground Truths:", fontsize=18, fontweight='bold', va='top')
        y_pos -= 0.08
        for i, ref in enumerate(s['references']):
            axes[1].text(0.1, y_pos, f"{i+1}. {ref}", fontsize=14, va='top', wrap=True)
            y_pos -= 0.06 * (1 + len(ref)//50)
            
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{category}_{img_name.replace('.jpg', '.png')}")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()

print(f"Saved {len(examples['Good']) + len(examples['Bad'])} simple plots to {output_dir}")
