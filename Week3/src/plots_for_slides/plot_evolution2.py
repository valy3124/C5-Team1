import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Settings
color_primary = '#023047'
color_accent = '#2d9e6b'

path = "/ghome/group01/C5/vali/C5-Team1/Week3/results/baseline/resnet18_lr0.0005_bs128_cmiqlnrw/captions_history.json"
with open(path, 'r') as f:
    data = json.load(f)

# Image to plot
img_name = "VizWiz_train_00006514.jpg"
image_path = f"/ghome/group01/C5/vali/C5-Team1/Week3/results/baseline/resnet18_lr0.0005_bs128_cmiqlnrw/visual_samples/{img_name}"

# Get truth
truth_refs = []
for r in data["1"]:
    if r['image_name'] == img_name:
        truth_refs = r['references']
        break

epochs_to_show = [1, 2, 4, 6, 8, 10, 12]

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('white')

# Grid layout
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])

# Left: Image and GT
ax_left = fig.add_subplot(gs[0, 0])
ax_left.axis('off')

img = mpimg.imread(image_path)
ax_left.imshow(img)
ax_left.set_title("Target Image: Turkey Franks", fontsize=20, fontweight='bold', color=color_primary, pad=20)

# Ground truths under image
gt_y = -30
ax_left.text(0, img.shape[0] + 50, "Ground Truth Captions:", fontsize=16, fontweight='bold', color=color_primary, va='top')
y_pos = img.shape[0] + 100
for i, t in enumerate(truth_refs[:3]):
    wrapped_text = "\n".join([t[j:j+50] for j in range(0, len(t), 50)])
    ax_left.text(0, y_pos, f"- {wrapped_text}", fontsize=14, color='#333333', va='top')
    y_pos += 50 * (len(wrapped_text.split('\n'))) + 30

# Right: Evolution
ax_right = fig.add_subplot(gs[0, 1])
ax_right.axis('off')

ax_right.text(0.05, 0.95, "Caption Evolution Across Epochs (Baseline Model)", fontsize=22, fontweight='bold', color=color_primary, va='top')

y_pos2 = 0.85
for ep in epochs_to_show:
    records = data[str(ep)]
    pred = ""
    for r in records:
        if r['image_name'] == img_name:
            pred = r['prediction']
            break
            
    # Highlight specific words or just show
    ax_right.text(0.05, y_pos2, f"Epoch {ep}:", fontsize=16, fontweight='bold', color=color_primary, va='top')
    
    # Text wrapping for prediction
    words = pred.split(' ')
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) < 65:
            current_line += word + " "
        else:
            lines.append(current_line)
            current_line = word + " "
    lines.append(current_line)
    
    pred_wrapped = "\n".join(lines)
    
    ax_right.text(0.20, y_pos2, pred_wrapped, fontsize=16, color='#444444', va='top', style='italic')
    
    y_pos2 -= 0.12

plt.tight_layout()
plt.savefig("/ghome/group01/C5/vali/C5-Team1/Week3/caption_evolution_franks.png", dpi=300, bbox_inches='tight')
print("Successfully generated caption_evolution_franks.png")
