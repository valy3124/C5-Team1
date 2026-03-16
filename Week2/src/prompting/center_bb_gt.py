import numpy as np
from typing import Any, Dict, List
import PIL.Image as Image
import pycocotools.mask as rletools

class CenterBBGTPromptStrategy:
    """
    Generates point prompts for SAM using the center of the Ground Truth bounding boxes,
    and also returns the bounding boxes for visualization purposes.
    """

    def generate_prompt(self, image: Image.Image, annotations: List[Any], **kwargs) -> Dict[str, Any]:
        points = []
        labels = []
        boxes_out = []
        
        for ann in annotations:
            # Format is [x1, y1, x2, y2] based on datasets.py
            x1, y1, x2, y2 = ann.bbox_xyxy
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            points.append([center_x, center_y])
            labels.append(1)
            boxes_out.append([x1, y1, x2, y2])
        
        # If not points, we just return empty arrays

        return {
            "type": "point_and_box",
            "points": np.array(points, dtype=np.float32),
            "point_labels": np.array(labels, dtype=np.int32),
            "boxes": np.array(boxes_out, dtype=np.float32)
        }
