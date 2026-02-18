"""
Ultralytics YOLO inference wrapper.

Provides the code for running inference
with Ultralytics YOLO models (task = object detection in this case).
"""

from __future__ import annotations
from typing import Literal, Optional, Dict, Any
import numpy as np
from PIL import Image
from ultralytics import YOLO


class UltralyticsYOLO:
    """
    Wrapper around Ultralytics YOLO models

    Args:
    -----
    weights : Optional[str]
        Path to model weights (can be the fine-tuned ones by us or the original)
    task : Literal["detect", "segment"]
        Type of task (in this project we use "detect")
    conf : float
        Confidence threshold
    iou : float
        IoU threshold for Non-Maximum Suppression
    device : str
        Device string ("0" for gpu)
    half : bool
        Whether to use FP16 inference
    """

    def __init__(
        self,
        weights: Optional[str] = None,
        task: Literal["detect", "segment"] = "detect",
        conf: float = 0.25,
        iou: float = 0.7,
        device: str = "0",
        half: bool = False,
    ) -> None:
        self.task = task
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half

        if weights is None:
            # TODO: XAVI
            raise ValueError("IMPLEMENT A DEFAULT FOR YOLO! (EASIER FOR INFERENCE IF WE DON'T HAVE IT DOWNLOADED)")
        
        self.model: YOLO = YOLO(weights)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on a single image

        Args
        -----
        image : PIL.Image.Image
            Input RGB image

        Returns
        -----
        Dict[str, Any]
            {
                "bboxes_xyxy": np.ndarray (N,4),
                "scores": np.ndarray (N,),
                "category_ids": np.ndarray (N,),
            }
        """

        results = self.model.predict(
            source=image,
            imgsz=max(image.size),
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        result = results[0]

        # No detections case
        if result.boxes is None or len(result.boxes) == 0:
            return {
                "bboxes_xyxy": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "category_ids": np.empty((0,), dtype=np.int64),
            }

        boxes_xyxy = (
            result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        )
        scores = (
            result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        )
        classes = (
            result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        )

        return {
            "bboxes_xyxy": boxes_xyxy,
            "scores": scores,
            "category_ids": classes
        }
