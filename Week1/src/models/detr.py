"""
Hugging Face DETR inference wrapper.

Provides the code for running inference with DETR
using the Hugging Face Transformers library.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import torch
import numpy as np
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

class HuggingFaceDETR:
    """
    Wrapper around Hugging Face DETR models

    Args:
    -----
    weights : Optional[str]
        If provided, path to model weights. Otherwise use default Hugging Face model
    conf : float
        Confidence threshold for filtering detections
    device : str
        Device string ("cuda" or "cpu")
    half : bool
        Whether to use FP16 (half-precision) inference.
        Note: If True, model is cast to .half() and input tensors are converted.
    """

    def __init__(
        self,
        weights: Optional[str] = None,
        conf: float = 0.5,
        device: str = "0",
        half: bool = False
    ) -> None:
        self.conf = conf
        self.device = device
        self.half = half

        # Default model: ResNet-50 backbone
        if weights is None:
            weights = "facebook/detr-resnet-50"

        self.processor = DetrImageProcessor.from_pretrained(weights)
        self.model = DetrForObjectDetection.from_pretrained(weights)
        self.model.to(self.device)
        
        # Apply Half Precision if requested
        if self.half:
            self.model.half()
            
        self.model.eval()

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
        # 1. Preprocessing
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Handle FP16 inputs if model is in half precision
        if self.half:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        # 2. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3. Post-processing (Image.size is WxH, but DETR expeects HxW)
        result = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=torch.tensor([image.size[::-1]]).to(self.device), 
            threshold=self.conf
        )[0]

        return {
            "bboxes_xyxy": result["boxes"].cpu().numpy().astype(np.float32),
            "scores": result["scores"].cpu().numpy().astype(np.float32),
            "category_ids": result["labels"].cpu().numpy().astype(np.int64),
        }