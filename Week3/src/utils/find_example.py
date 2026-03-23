import json

val_path = "/ghome/group01/C5/vali/C5-Team1/Week3/dataset/VizWiz/annotations/val.json"
with open(val_path, 'r') as f:
    val_data = json.load(f)

# format is usually standard coco: annotations have image_id, images have id and file_name
if "annotations" in val_data:
    from collections import defaultdict
    img_id_to_caps = defaultdict(list)
    for ann in val_data['annotations']:
        img_id_to_caps[ann['image_id']].append(ann['caption'])
    
    img_id_to_name = {img['id']: img['file_name'] for img in val_data['images']}
    
    count = 0
    for img_id, caps in img_id_to_caps.items():
        filt_caps = [c for c in caps if c != "Quality issues are too severe to recognize visual content."]
        if len(filt_caps) >= 5:
            print(f"Image: {img_id_to_name[img_id]}")
            for c in filt_caps:
                print(f" - {c}")
            print("---")
            count += 1
            if count > 5:
                break
