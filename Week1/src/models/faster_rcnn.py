from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import numpy as np
from PIL import Image

from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



class FasterRCNNModel:
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

    def prepare_finetune(self, num_classes, train_backbone):
        # Replace box predictor head
        in_feats = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)

        # Freeze/unfreeze backbone
        for p in self.model.backbone.parameters():
            p.requires_grad = train_backbone

        for p in self.model.roi_heads.box_predictor.parameters():
            p.requires_grad = True

        self.model.to(self.device)

    @torch.no_grad()
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
        # Since we might fien tune using this model, ensure we do not change eval state forever
        was_training = self.model.training
        self.model.eval()
        try:
            img = to_tensor(image).to(self.device)
            if self.half:
                # safer than pure f16, use autocast
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    result = self.model([img])[0]
            else:
                result = self.model([img])[0]
        finally:
            if was_training:
                self.model.train()
 
        return {
            "bboxes_xyxy": result["boxes"].detach().cpu().numpy().astype(np.float32),
            "scores": result["scores"].detach().cpu().numpy().astype(np.float32),
            "category_ids": result["labels"].detach().cpu().numpy().astype(np.int64),
        }