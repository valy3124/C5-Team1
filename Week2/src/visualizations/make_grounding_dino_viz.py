import os
import sys
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from datasets import DEART
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def add_title(img: Image.Image, title: str, bar_h: int = 40) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    tb = draw.textbbox((0, 0), title, font=font)
    draw.text(((w - (tb[2] - tb[0])) // 2, (bar_h - (tb[3] - tb[1])) // 2), title, fill="black", font=font)
    canvas.paste(img, (0, bar_h))
    return canvas

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading GroundingDINO...")
    dino_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(dino_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)
    model.eval()

    print("Loading DEArt...")
    ds = DEART(root=str(ROOT / "DEArt"), split="validation", ann_source="xml")
    
    out_dir = ROOT / "results_deart_dino_bboxes"
    out_dir.mkdir(exist_ok=True, parents=True)

    # 3 = crucifixion, 11 = shepherd
    target_classes = {3, 11}
    
    text_labels = "crucifixion . shepherd ."

    count = 0
    for i in range(len(ds)):
        image, anns, meta = ds[i]
        
        # Check if GT has our target classes
        has_target = any(ann.class_id in target_classes for ann in anns)
        if not has_target:
            continue
            
        print(f"Processing image {meta['index']}...")
        
        # Run GroundingDINO
        inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.25, text_threshold=0.25, target_sizes=[image.size[::-1]]
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        # Draw GroundingDINO boxes and labels
        det_img = image.copy()
        draw = ImageDraw.Draw(det_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except Exception:
            font = ImageFont.load_default()
            
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            
            # Draw box
            box_color = (255, 50, 50)
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=6)
            
            # Setup text
            text = f"{label} {score:.2f}"
            tb = draw.textbbox((0, 0), text, font=font)
            text_w = tb[2] - tb[0]
            text_h = tb[3] - tb[1]
            
            # Draw text background
            bg_rect = [x1, y1 - text_h - 4, x1 + text_w + 4, y1]
            # Ensure background doesn't go off top of image
            if bg_rect[1] < 0:
                bg_rect = [x1, y1, x1 + text_w + 4, y1 + text_h + 4]
                draw.rectangle(bg_rect, fill=box_color)
                draw.text((x1 + 2, y1 + 2), text, fill="white", font=font)
            else:
                draw.rectangle(bg_rect, fill=box_color)
                draw.text((x1 + 2, y1 - text_h - 2), text, fill="white", font=font)
            
        orig_panel = add_title(image.copy(), "Original")
        det_panel = add_title(det_img, f"GroundingDINO ({len(boxes)} boxes)")
        
        canvas = Image.new("RGB", (orig_panel.width * 2, orig_panel.height), "white")
        canvas.paste(orig_panel, (0, 0))
        canvas.paste(det_panel, (orig_panel.width, 0))
        
        canvas.save(out_dir / f"{meta['index']}_dino.png")
        count += 1
        
    print(f"Done! Saved {count} visualizations to {out_dir}")

if __name__ == '__main__':
    main()
