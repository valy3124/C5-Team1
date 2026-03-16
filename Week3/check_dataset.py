import os
import json

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'VizWiz')
img_train_path = os.path.join(base_path, 'images', 'train')
img_val_path = os.path.join(base_path, 'images', 'val')
img_test_path = os.path.join(base_path, 'images', 'test')

ann_train_path = os.path.join(base_path, 'annotations', 'train.json')
ann_val_path = os.path.join(base_path, 'annotations', 'val.json')

print("--- Image Counts ---")
try:
    print(f"Train images: {len(os.listdir(img_train_path))}")
except Exception as e:
    print(f"Train images error: {e}")

try:
    print(f"Validation images: {len(os.listdir(img_val_path))}")
except Exception as e:
    print(f"Validation images error: {e}")

try:
    print(f"Test images: {len(os.listdir(img_test_path))}")
except Exception as e:
    print(f"Test images error: {e}")

print("\n--- Annotation Counts ---")
try:
    with open(ann_train_path, 'r') as f:
        train_ann = json.load(f)
        print(f"Train Annotations Info:")
        print(f"  Images in JSON: {len(train_ann.get('images', []))}")
        print(f"  Captions in JSON: {len(train_ann.get('annotations', []))}")
except Exception as e:
    print(f"Train annotations error: {e}")

try:
    with open(ann_val_path, 'r') as f:
        val_ann = json.load(f)
        print(f"Validation Annotations Info:")
        print(f"  Images in JSON: {len(val_ann.get('images', []))}")
        print(f"  Captions in JSON: {len(val_ann.get('annotations', []))}")
except Exception as e:
    print(f"Validation annotations error: {e}")

