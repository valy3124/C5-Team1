from __future__ import annotations
from typing import Literal, Optional, Dict, Any, List, Union
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

class UltralyticsYOLO:
    """
    Wrapper around Ultralytics YOLO models optimized for high-speed inference.

    Args:
    -----
    weights : Optional[str]
        Path to model weights (Use .engine format for maximum FPS on RTX GPUs)
    conf : float
        Confidence threshold
    iou : float
        IoU threshold for Non-Maximum Suppression
    device : str
        Device string ("0" for gpu)
    half : bool
        Whether to use FP16 inference (Highly recommended for RTX 3090)
    map_coco : bool
        Whether to apply hardcoded class remapping to standard COCO-90 IDs
    """

    def __init__(
        self,
        weights: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.7,
        device: str = "0",
        half: bool = False,
        map_coco: bool = True,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half
        self.map_coco = map_coco

        # Default model
        if weights is None:
            weights = "yolov8s.pt" 
        
        self.model: YOLO = YOLO(weights)

    def predict(self, images: Union[Image.Image, List[Image.Image]]) -> tuple[Union[Dict[str, Any], List[Dict[str, Any]]], float]:
        """
        Run inference and return predictions alongside accurate inference time.
        """
        single_input = False
        if isinstance(images, Image.Image):
            images = [images]
            single_input = True
        
        results = self.model.predict(
            source=images,
            imgsz=640,
            iou=self.iou,
            conf=self.conf,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        outputs = []   
        total_time_ms = 0.0

        for result in results:
            frame_time_ms = result.speed['preprocess'] + result.speed['inference'] + result.speed['postprocess']
            total_time_ms += frame_time_ms

            if result.boxes is None or len(result.boxes) == 0:
                outputs.append({
                    "bboxes_xyxy": np.empty((0, 4), dtype=np.float32),
                    "scores": np.empty((0,), dtype=np.float32),
                    "category_ids": np.empty((0,), dtype=np.int64),
                })
                continue
        
            bboxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
            if self.map_coco:
                classes = np.where((classes == 0) | (classes == 2), classes + 1, 0)

            outputs.append({
                "bboxes_xyxy": bboxes,
                "scores": scores,
                "category_ids": classes,
            })

        final_output = outputs[0] if single_input else outputs
        
        avg_inference_time_sec = (total_time_ms / len(images)) / 1000.0

        return final_output, avg_inference_time_sec