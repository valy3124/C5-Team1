"""
Ultralytics YOLO inference wrapper.

Provides the code for running inference
with Ultralytics YOLO models
"""

from __future__ import annotations
from typing import Literal, Optional, Dict, Any, List, Union
import numpy as np
from PIL import Image
import torch
import time
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
    map_coco : bool
        Whether to apply hardcoded class remapping to standard COCO-90 IDs
        Disable this for fine-tuning on custom datasets.
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

        # Default model: YOLOv8 small pretrained on COCO
        if weights is None:
            weights = "yolov8s.pt"
        
        self.model: YOLO = YOLO(weights)

    def predict(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Run inference on a single image

        Args
        -----
        images : Union[PIL.Image.Image, List[PIL.Image.Image]]
            Input RGB image or list of RGB images.

        Returns
        -----
        List[Dict[str, Any]]
            One dictionary per image, each with:
            {
                "bboxes_xyxy": np.ndarray (Ni, 4),
                "scores": np.ndarray (Ni,),
                "category_ids": np.ndarray (Ni,),
            }
        """

        single_input = False
        if isinstance(images, Image.Image):
            images = [images]
            single_input = True
        
        #inference_time = self._benchmark_inference_time(images)
        inference_time = 0

        # It internally switches to eval mode.
        results = self.model.predict(
            source=images,
            imgsz=640,
            iou=self.iou,
            conf=self.conf,
            device=self.device,
            half=False,
            verbose=False,
        )

        outputs = []   

        # No detections case
        for result in results:
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

            # YOLO to standard COCO-90 Mapping (0=Person, 2=Car => 1=Person, 3=Car). Other classes are mapped to 0 (N/A)
            if self.map_coco:
                classes = np.where((classes == 0) | (classes == 2), classes + 1, 0)

            outputs.append({
                "bboxes_xyxy": bboxes,
                "scores": scores,
                "category_ids": classes,
            })

        final_output = outputs[0] if single_input else outputs
        # Return tuple: (predictions, inference_time)
        return final_output, inference_time

    def _benchmark_inference_time(self, sample_input: Any) -> float:
        """
        Calculates theoretical inference time using the actual first input tensor with warmup.
        Returns average inference time per frame in seconds.
        """
        if hasattr(self, "_cached_inference_time"):
            return self._cached_inference_time

        import time
        import torch
        
        device = self.device
        model = self.model.model
        was_training = model.training
        
        # device is usually "0" in ultralytics strings, but PyTorch `.to()` needs "cuda:0"
        device_pt = torch.device(f"cuda:{device}" if str(device).isdigit() else device)

        # Ultralytics models are often kept in CPU until predict is called. 
        # We must explicitly cast it for the benchmark dummy tensor.
        model.to(device_pt)
        model.eval()

        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        # Extract the first image from the input to use as our strictly fair dummy
        if isinstance(sample_input, list):
            sample_img = sample_input[0]
        else:
            sample_img = sample_input

        # Convert to tensor if it's a PIL Image (YOLO handles both)
        if isinstance(sample_img, Image.Image):
            dummy = to_tensor(sample_img.resize((640, 640))).unsqueeze(0).to(device_pt)
        elif isinstance(sample_img, torch.Tensor):
            dummy = sample_img.unsqueeze(0).to(device_pt) if sample_img.ndim == 3 else sample_img.to(device_pt)
        else:
            dummy = torch.randn(1, 3, 640, 640).to(device_pt)

        if self.half:
            dummy = dummy.half()

        try:
            with torch.no_grad():
                # Warmup
                for _ in range(20):
                    _ = model(dummy)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()

                # Benchmark
                for _ in range(100):
                    _ = model(dummy)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
            self._cached_inference_time = (end - start) / 100.0
            return self._cached_inference_time
        finally:
            if was_training:
                model.train()
