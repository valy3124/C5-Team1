import time
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamModel,
    SamProcessor,
)


class GroundedSamWrapper:
    """
    Grounded SAM: combines GroundingDINO (text-prompted object detector) with
    SAM (Segment Anything Model) to produce instance segmentation masks driven
    entirely by free-form text labels.

    Pipeline
    --------
    1. GroundingDINO receives the image + text labels and returns bounding boxes.
    2. The detected boxes are fed to SAM as ``"box"`` prompts.
    3. SAM returns one binary mask per box.

    The ``predict()`` interface is intentionally identical to ``SamWrapper`` so
    that ``run_inference.py`` works without any loop-level changes.
    """

    # Default Hugging Face model IDs
    DEFAULT_DINO_ID = "IDEA-Research/grounding-dino-tiny"
    DEFAULT_SAM_ID  = "facebook/sam-vit-base"

    def __init__(
        self,
        dino_model_id: str = DEFAULT_DINO_ID,
        sam_model_id: str  = DEFAULT_SAM_ID,
        box_threshold: float  = 0.35,
        text_threshold: float = 0.25,
        device: str = None,
    ):
        """
        Parameters
        ----------
        dino_model_id : str
            Hugging Face model ID for GroundingDINO.
        sam_model_id : str
            Hugging Face model ID for SAM.
        box_threshold : float
            Minimum detection confidence for a box to be kept.
        text_threshold : float
            Minimum token-level confidence used during label resolution.
        device : str, optional
            Target device string (``"cuda"``, ``"cpu"``, …).
            Auto-detected when *None*.
        """
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.box_threshold  = box_threshold
        self.text_threshold = text_threshold

        # ---- GroundingDINO ----
        print(f"Loading GroundingDINO ({dino_model_id}) on {self.device}...")
        self.dino_processor = AutoProcessor.from_pretrained(dino_model_id)
        self.dino_model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(dino_model_id)
            .to(self.device)
        )
        self.dino_model.eval()

        # ---- SAM ----
        print(f"Loading SAM ({sam_model_id}) on {self.device}...")
        self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
        self.sam_model = SamModel.from_pretrained(sam_model_id).to(self.device)
        self.sam_model.eval()

        print("GroundedSamWrapper loaded successfully!")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Image.Image,
        prompt_dict: Dict[str, Any],
    ) -> Tuple[List[torch.Tensor], torch.Tensor, float]:
        """
        Run Grounded SAM inference.
        """
        t_start = time.time()

        text_labels = prompt_dict.get("text", "person. car.")

        # Step 1: GroundingDINO: detect boxes from text
        boxes, det_scores = self._run_grounding_dino(image, text_labels)

        # Step 2: SAM: segment using detected boxes
        if len(boxes) == 0:
            h, w = image.size[1], image.size[0]
            empty_masks   = torch.zeros((1, 0, 3, h, w), dtype=torch.bool)
            empty_scores  = torch.zeros((1, 0, 1))
            return [empty_masks], empty_scores, time.time() - t_start

        masks_tensor, scores_tensor = self._run_sam(image, boxes, det_scores)

        inference_time = time.time() - t_start
        return [masks_tensor], scores_tensor, inference_time


    @torch.no_grad()
    def _run_grounding_dino(
        self,
        image: Image.Image,
        text_labels: str,
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Run GroundingDINO on *image* with *text_labels*.

        Returns
        -------
        boxes : list[list[float]]
            Detected boxes in ``[x_min, y_min, x_max, y_max]`` pixel coords.
        scores : list[float]
            Corresponding confidence scores.
        """
        try:
            inputs = self.dino_processor(
                images=image,
                text=[[lbl.strip() for lbl in text_labels.rstrip(".").split(".") if lbl.strip()]],
                return_tensors="pt",
            ).to(self.device)
        except Exception:
            inputs = self.dino_processor(
                images=image,
                text=text_labels,
                return_tensors="pt",
            ).to(self.device)

        outputs = self.dino_model(**inputs)

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )

        result = results[0]
        boxes  = result["boxes"].cpu().tolist()
        scores = result["scores"].cpu().tolist()

        return boxes, scores

    @torch.no_grad()
    def _run_sam(
        self,
        image: Image.Image,
        boxes: List[List[float]],
        det_scores: List[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run SAM with bounding-box prompts for each detected object.

        Returns
        -------
        masks_tensor : torch.Tensor
            Shape ``(1, N, 3, H, W)`` — SAM produces 3 mask candidates per
            box; the best candidate is selected by the inference loop using
            the IoU scores (identical to SamWrapper behaviour).
        scores_tensor : torch.Tensor
            Shape ``(1, N, 3)`` — IoU confidence per candidate mask.
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
        masks_tensor = masks[0].unsqueeze(0)

        scores_tensor = outputs.iou_scores.cpu()

        return masks_tensor, scores_tensor
