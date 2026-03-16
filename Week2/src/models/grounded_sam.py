import time
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List, Optional

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

    # Default label → COCO class-id mapping for KITTI-MOTS
    # COCO: 1=person, 3=car
    DEFAULT_LABEL_TO_CLASS_ID: Dict[str, int] = {
        # person synonyms (COCO id=1)
        "person":     1,
        "pedestrian": 1,
        "people":     1,
        "man":        1,
        "woman":      1,
        "human":      1,
        "cyclist":    1,
        "walker":     1,
        # car synonyms (COCO id=3)
        "car":        3,
        "vehicle":    3,
        "automobile": 3,
        "truck":      3,
        "van":        3,
        "sedan":      3,
        "suv":        3,
        "bus":        3,
    }

    def __init__(
        self,
        dino_model_id: str = DEFAULT_DINO_ID,
        sam_model_id: str  = DEFAULT_SAM_ID,
        box_threshold: float  = 0.35,
        text_threshold: float = 0.25,
        device: str = None,
        label_to_class_id: Optional[Dict[str, int]] = None,
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
        label_to_class_id : dict, optional
            Mapping from detected text label (lowercase) to integer class id.
            Used by ``predict_semantic()``.  Defaults to
            ``DEFAULT_LABEL_TO_CLASS_ID`` (KITTI-MOTS / COCO ids).
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
        self.label_to_class_id = (
            label_to_class_id
            if label_to_class_id is not None
            else dict(self.DEFAULT_LABEL_TO_CLASS_ID)
        )

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
        boxes, det_scores, _det_labels = self._run_grounding_dino(image, text_labels)

        # Step 2: SAM: segment using detected boxes
        if len(boxes) == 0:
            h, w = image.size[1], image.size[0]
            empty_masks   = torch.zeros((1, 0, 3, h, w), dtype=torch.bool)
            empty_scores  = torch.zeros((1, 0, 1))
            return [empty_masks], empty_scores, time.time() - t_start

        masks_tensor, scores_tensor = self._run_sam(image, boxes, det_scores)

        inference_time = time.time() - t_start
        return [masks_tensor], scores_tensor, inference_time

    def predict_semantic(
        self,
        image: Image.Image,
        prompt_dict: Dict[str, Any],
    ) -> Tuple[np.ndarray, float]:
        """
        Run Grounded SAM and produce a **semantic segmentation map**.

        Unlike ``predict()``, this method merges all per-instance masks into a
        single ``(H, W)`` integer array where each pixel holds the class id
        (from ``label_to_class_id``) of the highest-confidence detection that
        covers it (background = 0).

        Returns
        -------
        semantic_map : np.ndarray, shape (H, W), dtype int32
        inference_time : float
        """
        t_start = time.time()
        text_labels = prompt_dict.get("text", "person. car.")
        h, w = image.size[1], image.size[0]
        semantic_map = np.zeros((h, w), dtype=np.int32)

        boxes, det_scores, det_labels = self._run_grounding_dino(image, text_labels)

        if len(boxes) == 0:
            return semantic_map, time.time() - t_start

        masks_tensor, scores_tensor = self._run_sam(image, boxes, det_scores)
        # masks_tensor: (1, N, 3, H, W) — squeeze batch dim
        masks_out  = masks_tensor.squeeze(0)          # (N, 3, H, W)
        scores_out = scores_tensor.squeeze(0)         # (N, 3)
        if scores_out.dim() == 1:
            scores_out = scores_out.unsqueeze(0)

        # Select best mask candidate per detection and pick its class id.
        # Process detections from lowest to highest confidence so that the
        # most-confident mask wins on overlapping pixels.
        best_idx_per_det   = scores_out.argmax(dim=-1)          # (N,)
        best_score_per_det = scores_out[torch.arange(len(boxes)), best_idx_per_det]  # (N,)
        order = torch.argsort(best_score_per_det)               # ascending

        for j in order.tolist():
            class_id = self._resolve_class_id(det_labels[j])
            if class_id == 0:
                continue
            mask_np = masks_out[j, int(best_idx_per_det[j])].cpu().numpy()  # (H, W) bool
            semantic_map[mask_np] = class_id

        return semantic_map, time.time() - t_start

    def predict_semantic_open(
        self,
        image: Image.Image,
        prompt_dict: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[int, str], float]:
        """
        Open-vocabulary semantic segmentation: segment *any* detected class
        without requiring a fixed ``label_to_class_id`` mapping.

        Each unique detected label is assigned a sequential integer id starting
        from 1.  Background = 0.  The returned ``id_to_label`` dict lets callers
        build a dynamic legend / colour palette.

        Returns
        -------
        semantic_map : np.ndarray, shape (H, W), dtype int32
        id_to_label : dict[int, str]   e.g. {1: "person", 2: "car", 3: "tree"}
        inference_time : float
        """
        t_start = time.time()
        text_labels = prompt_dict.get("text", "person. car.")
        h, w = image.size[1], image.size[0]
        semantic_map = np.zeros((h, w), dtype=np.int32)

        boxes, det_scores, det_labels = self._run_grounding_dino(image, text_labels)

        if len(boxes) == 0:
            return semantic_map, {}, time.time() - t_start

        # Build a dynamic label → sequential id mapping
        label_to_id: Dict[str, int] = {}
        id_to_label: Dict[int, str] = {}
        next_id = 1
        for lbl in det_labels:
            key = lbl.lower().strip().rstrip(".")
            if key not in label_to_id:
                label_to_id[key] = next_id
                id_to_label[next_id] = key
                next_id += 1

        masks_tensor, scores_tensor = self._run_sam(image, boxes, det_scores)
        masks_out  = masks_tensor.squeeze(0)   # (N, 3, H, W)
        scores_out = scores_tensor.squeeze(0)  # (N, 3)
        if scores_out.dim() == 1:
            scores_out = scores_out.unsqueeze(0)

        best_idx_per_det   = scores_out.argmax(dim=-1)
        best_score_per_det = scores_out[torch.arange(len(boxes)), best_idx_per_det]
        order = torch.argsort(best_score_per_det)   # ascending: best overwrites

        for j in order.tolist():
            key      = det_labels[j].lower().strip().rstrip(".")
            class_id = label_to_id.get(key, 0)
            if class_id == 0:
                continue
            mask_np = masks_out[j, int(best_idx_per_det[j])].cpu().numpy()
            semantic_map[mask_np] = class_id

        return semantic_map, id_to_label, time.time() - t_start

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_class_id(self, label: str) -> int:
        """Map a detected text label to its integer class id (0 = unknown)."""
        label_lower = label.lower().strip().rstrip(".")
        # Exact match first
        if label_lower in self.label_to_class_id:
            return self.label_to_class_id[label_lower]
        # Substring match
        for key, cid in self.label_to_class_id.items():
            if key in label_lower or label_lower in key:
                return cid
        return 0


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

        result  = results[0]
        boxes   = result["boxes"].cpu().tolist()
        scores  = result["scores"].cpu().tolist()
        labels  = result.get("labels", ["" for _ in boxes])

        return boxes, scores, labels

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
        # Note: det_scores unused here — SAM's iou_scores are used for ranking
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
