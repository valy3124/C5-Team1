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
    weights : str
        Path to model weights (can be the fine-tuned ones by us or the original) or model name
    task : Literal["detect", "segment"]
        Type of task (in this project we will use "detect")
    imgsz : int
        Inference image size
    conf : float
        Confidence threshold
    iou : float
        IoU threshold for Non-Maximum Suppression
    device : str
        Device string ("0" for gpu)
    half : bool
        Whether to use FP16 inference
    max_det : int
        Maximum number of detections per image
    """

    def __init__(
        self,
        weights: str,
        task: Literal["detect", "segment"] = "detect",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str = "0",
        half: bool = False,
        max_det: int = 300,
    ) -> None:
        self.task = task
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half
        self.max_det = max_det

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
                "boxes_xyxy": np.ndarray (N,4),
                "scores": np.ndarray (N,),
                "classes": np.ndarray (N,),
                "masks": Optional[np.ndarray] (N,H,W) (always none in this project)
            }
        """

        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            max_det=self.max_det,
            verbose=False,
        )

        result = results[0]

        # No detections case
        if result.boxes is None or len(result.boxes) == 0:
            return {
                "boxes_xyxy": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "classes": np.empty((0,), dtype=np.int64),
                "masks": None,
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

        masks: Optional[np.ndarray] = None

        return {
            "boxes_xyxy": boxes_xyxy,
            "scores": scores,
            "classes": classes,
            "masks": masks,
        }
