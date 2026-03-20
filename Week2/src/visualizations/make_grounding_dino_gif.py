import os
import sys
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from datasets import DEART
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def draw_boxes(image: Image.Image, boxes, scores, labels, thresh: float) -> Image.Image:
    det_img = image.copy()
    draw = ImageDraw.Draw(det_img)
    try:
        font_box = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 46)
    except Exception:
        font_box = ImageFont.load_default()
        font_title = ImageFont.load_default()
        
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        
        box_color = (255, 50, 50)
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=6)
        
        text = f"{label} {score:.2f}"
        tb = draw.textbbox((0, 0), text, font=font_box)
        text_w = tb[2] - tb[0]
        text_h = tb[3] - tb[1]
        
        bg_rect = [x1, y1 - text_h - 4, x1 + text_w + 4, y1]
        if bg_rect[1] < 0:
            bg_rect = [x1, y1, x1 + text_w + 4, y1 + text_h + 4]
            draw.rectangle(bg_rect, fill=box_color)
            draw.text((x1 + 2, y1 + 2), text, fill="white", font=font_box)
        else:
            draw.rectangle(bg_rect, fill=box_color)
            draw.text((x1 + 2, y1 - text_h - 2), text, fill="white", font=font_box)
            
    # Add the threshold parameters directly onto the top-left of the image
    param_text = f"box_threshold: {thresh:.2f}"
    tb = draw.textbbox((0, 0), param_text, font=font_title)
    text_w = tb[2] - tb[0]
    text_h = tb[3] - tb[1]
    
    # Draw a semi-transparent black background for the parameter text
    draw.rectangle([10, 10, 10 + text_w + 20, 10 + text_h + 20], fill=(0, 0, 0, 180))
    draw.text((20, 20), param_text, fill="white", font=font_title)
            
    return det_img

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading GroundingDINO...")
    dino_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(dino_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)
    model.eval()

    print("Loading DEArt...")
    ds = DEART(root=str(ROOT / "DEArt"), split="validation", ann_source="xml")
    
    out_dir = ROOT / "results_deart_dino_gifs"
    out_dir.mkdir(exist_ok=True, parents=True)

    text_labels = "crucifixion ."

    target_img = None
    target_meta = None
    target_anns = None
    
    for i in range(len(ds)):
        image, anns, meta = ds[i]
        if any(ann.class_id == 3 for ann in anns): # 3 = crucifixion
            target_img = image.convert("RGBA") # Ensure we can overlay transparent boxes
            target_meta = meta
            target_anns = anns
            break
            
    if target_img is None:
        print("Could not find a crucifixion image!")
        return
        
    print(f"Generating frames for image {target_meta['index']}...")
    
    inputs = processor(images=target_img.convert("RGB"), text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        
    frames = []
    
    # Scale down the image slightly to make the GIF smaller (max width 1200)
    max_w = 1200
    if target_img.width > max_w:
        ratio = max_w / target_img.width
        new_size = (max_w, int(target_img.height * ratio))
        target_img = target_img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Need to re-run bounds with new scale so boxes match
        inputs = processor(images=target_img.convert("RGB"), text=text_labels, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
    
    thresholds = np.arange(0.10, 0.60, 0.05)
    
    for thresh in thresholds:
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=thresh, text_threshold=0.25, target_sizes=[target_img.size[::-1]]
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]
        
        det_panel = draw_boxes(target_img, boxes, scores, labels, thresh)
        
        # Convert back to RGB for gif saving
        frames.append(np.array(det_panel.convert("RGB")))
        
    frames.extend(frames[::-1])
    
    gif_path = out_dir / f"crucifixion_threshold_sweep_{target_meta['index']}.gif"
    print(f"Saving GIF to {gif_path}...")
    # Add optimize=True to heavily reduce filesize
    imageio.mimsave(gif_path, frames, fps=2, loop=0, optimize=True)
    print("Done!")

if __name__ == '__main__':
    main()
