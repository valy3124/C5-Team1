import torch
from transformers import SamModel, SamProcessor
import numpy as np

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base")

image = np.zeros((224, 224, 3), dtype=np.uint8)

# Correct format for HF SAM: (batch_size, point_batch_size, num_points_per_image, 2)
# We want: 1 image, 3 objects (point_batch_size=3), 1 point per object (num_points_per_image=1)
points = [[[[10, 10]], [[50, 50]], [[100, 100]]]]
labels = [[[[1]], [[1]], [[1]]]]

print("Testing points shape...")
try:
    inputs = processor(image, input_points=points, input_labels=labels, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    print("Masks output shapes:", outputs.pred_masks.shape)
    print("Scores output shape:", outputs.iou_scores.shape)
except Exception as e:
    print("Error:", e)
