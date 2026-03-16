"""
YoloSamWrapper
==============
Two-stage instance segmentation pipeline:
  1. YOLOv10 (fine-tuned or pretrained COCO) → bounding boxes
  2. SAM (Segment Anything) → one binary mask per box

The ``predict()`` return signature is identical to ``GroundedSamWrapper``
(4 values: masks, iou_scores, inference_time, boxes) so ``run_inference.py``
works without any loop changes.

Usage
-----
- Fine-tuned weights:
    YoloSamWrapper(yolo_weights="Week2/yolo_sweep_7q7hwud0/best_model.pt", ...)
- Pretrained YOLOv10b (COCO):
    YoloSamWrapper(yolo_weights="pretrained", ...)  # or None
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor
from ultralytics import YOLO


# COCO class indices → KITTI-MOTS COCO category ids (person=1, car=3)
# Pretrained YOLOv10b uses COCO classes: person=0, car=2
_PRETRAINED_CLS_TO_COCO_CAT = {0: 1, 2: 3}  # person→1, car→3

# Fine-tuned on KITTI-MOTS yolo_dataset (data.yaml: 0=Car, 1=Pedestrian)
_FINETUNED_CLS_TO_COCO_CAT  = {0: 3, 1: 1}  # Car→3, Pedestrian→1

# COCO class indices that correspond to KITTI-MOTS classes (pretrained filter)
_KITTI_COCO_CLASSES = set(_PRETRAINED_CLS_TO_COCO_CAT.keys())
# Default pretrained model tag understood by Ultralytics
_PRETRAINED_WEIGHTS = "yolov10b.pt"


class YoloSamWrapper:
    """
    Runs YOLOv10 detection followed by SAM segmentation.

    Parameters
    ----------
    yolo_weights : str or None
        Path to a ``.pt`` file with fine-tuned YOLO weights, or ``"pretrained"``
        / ``None`` to use the official YOLOv10b COCO checkpoint.
    sam_model_id : str
        Hugging Face model ID for SAM.
    conf_threshold : float
        Minimum YOLO confidence score for a box to be forwarded to SAM.
    device : str or None
        Target device. Auto-detected when *None*.
    """

    DEFAULT_SAM_ID = "facebook/sam-vit-base"

    def __init__(
        self,
        yolo_weights: Optional[str] = None,
        sam_model_id: str = DEFAULT_SAM_ID,
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        # ── Device ──────────────────────────────────────────────────────────
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.conf_threshold = conf_threshold

        # ── YOLO ────────────────────────────────────────────────────────────
        if yolo_weights is None or str(yolo_weights).lower() == "pretrained":
            yolo_ckpt = _PRETRAINED_WEIGHTS
            self._filter_classes = True   # pretrained → filter to person+car
            print(f"Loading pretrained YOLOv10b ({yolo_ckpt})...")
        else:
            yolo_ckpt = str(yolo_weights)
            self._filter_classes = False  # fine-tuned already outputs 2 classes
            print(f"Loading fine-tuned YOLO from {yolo_ckpt}...")

        self.yolo = YOLO(yolo_ckpt)
        self.yolo.to(self.device)

        # ── SAM ─────────────────────────────────────────────────────────────
        print(f"Loading SAM ({sam_model_id}) on {self.device}...")
        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
        self.sam_model = SamModel.from_pretrained(sam_model_id).to(self.device)
        self.sam_model.eval()

        print("YoloSamWrapper loaded successfully!")

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        image: Image.Image,
        prompt_dict: Dict[str, Any],
    ) -> Tuple[List[torch.Tensor], torch.Tensor, float, List[List[float]], List[int]]:
        """
        Run YOLO → SAM pipeline on *image*.

        ``prompt_dict`` is accepted for API compatibility but **ignored** —
        the detections come entirely from YOLO.

        Returns
        -------
        ``([masks_tensor], scores_tensor, inference_time, yolo_boxes, coco_cat_ids)``
        where ``coco_cat_ids[i]`` is the COCO category id for the i-th detected box.
        """
        t_start = time.time()

        boxes, coco_cat_ids = self._run_yolo(image)

        if len(boxes) == 0:
            h, w = image.size[1], image.size[0]
            empty_masks  = torch.zeros((1, 0, 3, h, w), dtype=torch.bool)
            empty_scores = torch.zeros((1, 0, 3))
            return [empty_masks], empty_scores, time.time() - t_start, [], []

        masks_tensor, scores_tensor = self._run_sam(image, boxes)

        return [masks_tensor], scores_tensor, time.time() - t_start, boxes, coco_cat_ids

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _run_yolo(self, image: Image.Image) -> Tuple[List[List[float]], List[int]]:
        """
        Run YOLO on *image* and return:
          - boxes: [[x1,y1,x2,y2], ...] in pixel coordinates
          - coco_cat_ids: COCO category id per box (person=1, car=3)
        """
        cls_to_cat = (
            _PRETRAINED_CLS_TO_COCO_CAT
            if self._filter_classes
            else _FINETUNED_CLS_TO_COCO_CAT
        )

        results = self.yolo.predict(
            source=image,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )

        boxes: List[List[float]] = []
        coco_cat_ids: List[int] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                # Pretrained: keep only person+car; fine-tuned: keep all
                if self._filter_classes and cls_id not in _KITTI_COCO_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                boxes.append([x1, y1, x2, y2])
                coco_cat_ids.append(cls_to_cat.get(cls_id, 1))

        return boxes, coco_cat_ids

    @torch.no_grad()
    def _run_sam(
        self,
        image: Image.Image,
        boxes: List[List[float]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run SAM with *boxes* as bounding-box prompts.

        Returns
        -------
        masks_tensor : torch.Tensor  shape (1, N, 3, H, W)
        scores_tensor : torch.Tensor shape (1, N, 3)
        """
        input_boxes = [[[box] for box in boxes]]

        inputs = self.sam_processor(
            image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.sam_model(**inputs)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        masks_tensor = masks[0].unsqueeze(0)          # (1, N, 3, H, W)
        scores_tensor = outputs.iou_scores.cpu()      # (1, N, 3)

        return masks_tensor, scores_tensor
