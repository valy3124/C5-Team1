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

        import os
        
        # Default model: ResNet-50 backbone
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
        
        # Apply Half Precision if requested
        if self.half:
            self.model.half()

    def predict(self, image: Union[Image.Image, List[Image.Image]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
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

        single_input = False
        if isinstance(image, Image.Image):
            image = [image]
            single_input = True

        inference_time = self._benchmark_inference_time(image)

        # Since we might fine tune using this model, ensure we do not change eval state forever
        was_training = self.model.training
        self.model.eval()
        try:
            # 1. Preprocessing (NOT counted in inference time)
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Handle FP16 inputs if model is in half precision
            if self.half:
                inputs["pixel_values"] = inputs["pixel_values"].half()

            # 2. Inference
            with torch.no_grad():
                outputs = self.model(**inputs)

        finally:
            if was_training:
                self.model.train()

        # 3. Post-processing (Image.size is WxH, but DETR expects HxW) (NOT counted)
        target_sizes = torch.tensor([img.size[::-1] for img in image]).to(self.device)

        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.conf,
        )

        outputs_list: List[Dict[str, Any]] = []

        # The mapping done in dataset.py is already aligned with DETR model
        for result in results:
            outputs_list.append({
                "bboxes_xyxy": result["boxes"].detach().cpu().numpy().astype(np.float32),
                "scores": result["scores"].detach().cpu().numpy().astype(np.float32),
                "category_ids": result["labels"].detach().cpu().numpy().astype(np.int64),
            })

        final_output = outputs_list[0] if single_input else outputs_list
        return final_output, inference_time

    def _benchmark_inference_time(self, sample_input: Any) -> float:
        """
        Calculates theoretical inference time using the actual first input tensor with warmup.
        Returns average inference time per frame in seconds.
        """
        if hasattr(self, "_cached_inference_time"):
            return self._cached_inference_time

        import time
        
        was_training = self.model.training
        self.model.eval()

        if isinstance(sample_input, list):
            sample_img = sample_input[0]
        else:
            sample_img = sample_input
            
        # Ensure it's PIL for DETR processor
        if isinstance(sample_img, torch.Tensor):
            from torchvision.transforms.functional import to_pil_image
            sample_img = to_pil_image(sample_img)

        dummy_inputs = self.processor(images=sample_img, return_tensors="pt")
        dummy_inputs = {k: v.to(self.device) for k, v in dummy_inputs.items()}
        if self.half:
            dummy_inputs["pixel_values"] = dummy_inputs["pixel_values"].half()

        try:
            with torch.no_grad():
                # Warmup
                for _ in range(20):
                    _ = self.model(**dummy_inputs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()

                # Benchmark
                for _ in range(100):
                    _ = self.model(**dummy_inputs)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
            self._cached_inference_time = (end - start) / 100.0
            return self._cached_inference_time
        finally:
            if was_training:
                self.model.train()