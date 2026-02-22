from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import numpy as np
from PIL import Image
import torchvision

from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



class FasterRCNNModel(torch.nn.Module):
    """
    Wrapper around Torchvision FasterRCNN models.

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
        device: str = "cuda:0",
        half: bool = False
    ) -> None:
        super().__init__()
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half

        if weights is None:
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT

        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights,
            box_score_thresh=self.conf,
            box_nms_thresh=self.iou
        ).to(self.device)

        # Will likely fail if not satisified
        if self.half:
            if "cuda" not in self.device:
                raise ValueError("half=True requires CUDA.")
            self.model = self.model.half()

    def prepare_finetune(self, num_classes, freeze_strategy=1):
        """
        Prepare the model for fine-tuning by replacing the head and applying freezing strategies.
        
        freeze_strategy:
            1: Freeze Backbone + RPN + ROI Heads (Train ONLY the new Box Predictor)
            2: Freeze Backbone + RPN. (Train ALL ROI Heads + Box Predictor)
            3: Freeze Backbone. (Train RPN + ROI Heads + Box Predictor)
            4: Full Training. (Train EVERYTHING)
        """
        # 1. First, freeze EVERYTHING by default for safety
        for p in self.model.parameters():
            p.requires_grad = False
            
        # 2. Re-initialize the box predictor head (creates new layers with requires_grad=True)
        in_feats = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)

        # 3. Apply Unfreezing Strategy
        if freeze_strategy >= 2:
            # Unfreeze the rest of the ROI heads (feature extractors in the neck)
            for p in self.model.roi_heads.parameters():
                p.requires_grad = True
                
        if freeze_strategy >= 3:
            # Unfreeze the Region Proposal Network (RPN)
            for p in self.model.rpn.parameters():
                p.requires_grad = True
                
        if freeze_strategy == 4:
            # Unfreeze the Backbone
            for p in self.model.backbone.parameters():
                p.requires_grad = True

        self.model.to(self.device)

    def forward(self, images, targets=None):
        return self.model(images, targets)

    @torch.no_grad()
    def predict(self, images: Union[Image.Image, List[Image.Image], torch.Tensor, List[torch.Tensor]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run inference on a single image or a list of images

        Args
        -----
        images : PIL.Image.Image, List[PIL.Image.Image], torch.Tensor, or List[torch.Tensor]
            Input RGB image(s) or tensor(s)

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
        if isinstance(images, (Image.Image, torch.Tensor)):
            images = [images]
            single_input = True

        # Preserve original training state
        was_training = self.model.training
        self.model.eval()

        try:
            tensors = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    tensors.append(img.to(self.device))
                else:
                    tensors.append(to_tensor(img).to(self.device))

            if self.half:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    results = self.model(tensors)
            else:
                results = self.model(tensors)

        finally:
            if was_training:
                self.model.train()

        outputs = []
        for result in results:
            outputs.append({
                "bboxes_xyxy": result["boxes"].detach().cpu().numpy().astype(np.float32),
                "scores": result["scores"].detach().cpu().numpy().astype(np.float32),
                "category_ids": result["labels"].detach().cpu().numpy().astype(np.int64),
            })

        if single_input:
            return outputs[0]

        return outputs