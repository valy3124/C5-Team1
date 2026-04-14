import os
import json
import textwrap
import argparse
from PIL import Image, ImageDraw, ImageFont

def explore_vizwiz(dataset_dir, output_dir, max_images=100, split="val"):
    """
    Goes over VizWiz dataset and saves images with their GT captions in a folder
    for general exploration.
    
    Args:
        dataset_dir (str): Path to VizWiz dataset folder.
        output_dir (str): Path to output folder.
        max_images (int): Maximum number of images to process.
        split (str): 'train' or 'val'.
    """
    images_dir = os.path.join(dataset_dir, "images", split)
    annotations_file = os.path.join(dataset_dir, "annotations", f"{split}.json")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Loading annotations from {annotations_file}...")
    try:
        with open(annotations_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotations file not found at {annotations_file}")
        return
        
    print("Mapping annotations...")
    img_id_to_caps = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in img_id_to_caps:
            img_id_to_caps[img_id] = []
        img_id_to_caps[img_id].append(ann.get("caption", ""))
        
    processed_count = 0
    


    html_lines = ["<html><head><style>img {max-width: 400px;}</style></head><body style='font-family: sans-serif;'>"]
    html_lines.append(f"<h1>VizWiz {split.capitalize()} Exploration</h1>")
    
    print(f"Processing up to {max_images} images...")
    
    for img_info in data.get("images", []):
        if max_images > 0 and processed_count >= max_images:
            break
            
        img_id = img_info.get("id")
        file_name = img_info.get("file_name")
        if not file_name:
            continue
            
        img_path = os.path.join(images_dir, file_name)
        if not os.path.exists(img_path):
            continue
            
        captions = img_id_to_caps.get(img_id, [])
        if not captions:
            captions = ["No caption available"]
            
        out_img_path = os.path.join(output_dir, file_name)
        
        try:
            img = Image.open(img_path).convert("RGB")
            
            # We will pad the bottom and draw captions there
            w, h = img.size
            
            font_size = max(14, int(w * 0.04))
            try:
                dyn_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except IOError:
                dyn_font = ImageFont.load_default()
                
            # Average char width estimation
            char_width = max(6, int(font_size * 0.55))
            chars_per_line = max(10, (w - 20) // char_width)
            
            wrapped_captions = []
            for cap in captions:
                wrapped_captions.extend(textwrap.wrap(cap, width=chars_per_line))
                
            line_spacing = int(font_size * 1.2)
            padding = 20 + line_spacing * len(wrapped_captions)
            
            new_img = Image.new("RGB", (w, h + padding), (255, 255, 255))
            new_img.paste(img, (0, 0))
            
            draw = ImageDraw.Draw(new_img)
            y_text = h + 10
            for line in wrapped_captions:
                draw.text((10, y_text), line, font=dyn_font, fill=(0, 0, 0))
                y_text += line_spacing
                
            new_img.save(out_img_path)
            
            # HTML generation
            html_lines.append("<div>")
            html_lines.append(f"<h3>{file_name}</h3>")
            html_lines.append(f"<img src='{file_name}' />")
            html_lines.append("<ul>")
            for cap in captions:
                html_lines.append(f"<li>{cap}</li>")
            html_lines.append("</ul>")
            html_lines.append("</div><hr>")
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count} images...")
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    html_lines.append("</body></html>")
    html_path = os.path.join(output_dir, f"explore_{split}.html")
    with open(html_path, "w") as f:
        f.write("\n".join(html_lines))
        
    print(f"Done! Processed {processed_count} images. Results saved in {output_dir}")
    print(f"You can view the HTML summary report at {html_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore VizWiz Images and GT captions")
    parser.add_argument("--dataset_dir", type=str, default="/ghome/group01/C5/benet/C5-Team1/Week5/VizWiz", help="Path to VizWiz dataset")
    parser.add_argument("--output_dir", type=str, default="/ghome/group01/C5/benet/C5-Team1/Week5/Exploration_VizWiz", help="Path to output exploration folder")
    parser.add_argument("--max_images", type=int, default=100, help="Max images to explore. Set to 0 or -1 to process all.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to explore")
    
    args = parser.parse_args()
    explore_vizwiz(args.dataset_dir, args.output_dir, args.max_images, args.split)
