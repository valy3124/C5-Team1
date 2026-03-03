from __future__ import annotations
from typing import Dict, Any, Optional, List, Union
import torch
import numpy as np
import time
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
        Device string ("cuda" or "0")
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
        self.device = f"cuda:{device}" if str(device).isdigit() else device
        self.half = half

        import os
        
        if weights is None or weights == "DEFAULT":
            hf_source = "facebook/detr-resnet-50"
            is_local_pth = False
        else:
            is_local_pth = os.path.isfile(weights) and (weights.endswith(".pth") or weights.endswith(".pt"))
            hf_source = "facebook/detr-resnet-50" if is_local_pth else weights

        self.processor = DetrImageProcessor.from_pretrained(hf_source)
        
        self.model = DetrForObjectDetection.from_pretrained(
            hf_source,
            ignore_mismatched_sizes=True if is_local_pth else False
        )
        
        if is_local_pth:
            state_dict = torch.load(weights, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]

            cls_weight = state_dict.get("class_labels_classifier.weight")
            if cls_weight is not None:
                num_classes_in_ckpt = cls_weight.shape[0]
                d_model = self.model.config.d_model
                self.model.class_labels_classifier = torch.nn.Linear(d_model, num_classes_in_ckpt)
            
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        
        if self.half:
            self.model.half()

    def predict(self, image: Union[Image.Image, List[Image.Image]]) -> tuple[Union[Dict[str, Any], List[Dict[str, Any]]], float]:
        """
        Run inference on a single image or list of images and return End-to-End time.
        """
        single_input = False
        if isinstance(image, Image.Image):
            image = [image]
            single_input = True

        was_training = self.model.training
        self.model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.half:
                inputs["pixel_values"] = inputs["pixel_values"].half()

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.tensor([img.size[::-1] for img in image]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self.conf,
            )

        finally:
            if was_training:
                self.model.train()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        avg_inference_time_sec = (end_time - start_time) / len(image)

        outputs_list: List[Dict[str, Any]] = []

        for result in results:
            outputs_list.append({
                "bboxes_xyxy": result["boxes"].detach().cpu().numpy().astype(np.float32),
                "scores": result["scores"].detach().cpu().numpy().astype(np.float32),
                "category_ids": result["labels"].detach().cpu().numpy().astype(np.int64),
            })

        final_output = outputs_list[0] if single_input else outputs_list
        return final_output, avg_inference_time_sec