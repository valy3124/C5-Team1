"""
Ultralytics YOLO inference wrapper.

Provides the code for running inference
with Ultralytics YOLO models
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
        conf: float = 0.25,
        iou: float = 0.7,
        device: str = "0",
        half: bool = False,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half

        # Default model: YOLOv8 small pretrained on COCO
        if weights is None:
            weights = "yolov8s.pt"
        
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

        result = self.model.predict(
            source=image,
            imgsz=max(image.size),
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        # No detections case
        if result.boxes is None or len(result.boxes) == 0:
            return {
                "bboxes_xyxy": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "category_ids": np.empty((0,), dtype=np.int64),
            }
        
        bboxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)

        # YOLO to standard COCO-90 Mapping (0=Person, 2=Car => 1=Person, 3=Car). Other classes are mapped to 0 (N/A)
        classes = np.where((classes == 0) | (classes == 2), classes + 1, 0)

        return {
            "bboxes_xyxy": bboxes,
            "scores": scores,
            "category_ids": classes,
        }
