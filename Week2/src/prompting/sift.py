import numpy as np
import cv2
from typing import Any, Dict, List
import PIL.Image as Image

class SiftPromptStrategy:
    """
    Uses SIFT keypoints extracted from the entire image as point prompts for SAM.
    """
    def generate_prompt(self, image: Image.Image, annotations: List[Any], **kwargs) -> Dict[str, Any]:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        keypoints = sift.detect(img_cv, None)

        if len(keypoints) == 0:
            h, w = img_cv.shape
            points = np.array([[w / 2, h / 2]], dtype=np.float32)
        else:
            keypoints = sorted(keypoints, key=lambda kp: -kp.response)

            # Limit number of points
            max_points = kwargs.get("max_points", 20)
            keypoints = keypoints[:max_points]
            points = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        # All prompts are positive
        labels = np.ones((len(points),), dtype=np.int32)

        return {
            "type": "point",
            "points": points,
            "point_labels": labels
        }