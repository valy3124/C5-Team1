import numpy as np
from typing import Any, Dict, List
import PIL.Image as Image

class GridPromptStrategy:
    """
    Generates a uniform grid of point prompts for SAM over the entire image.
    No bounding boxes or annotations are used.
    """

    def generate_prompt(self, image: Image.Image, annotations: List[Any], **kwargs) -> Dict[str, Any]:
        w, h = image.size
        points_per_side = kwargs.get("points_per_side", 10)
        offset_x = w / (2 * points_per_side)
        offset_y = h / (2 * points_per_side)
        x_coords = np.linspace(offset_x, w - offset_x, points_per_side)
        y_coords = np.linspace(offset_y, h - offset_y, points_per_side)
        xv, yv = np.meshgrid(x_coords, y_coords)
        points = np.stack([xv.ravel(), yv.ravel()], axis=-1).astype(np.float32)
        
        # All points are positive
        labels = np.ones((len(points),), dtype=np.int32)

        return {
            "type": "point",
            "points": points,
            "point_labels": labels
        }