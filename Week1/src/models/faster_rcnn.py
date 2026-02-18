from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataclasses import dataclass
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



# Albumentations transforms for Faster-RCNN
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.02, scale_limit=0.10, rotate_limit=5,
                border_mode=0, p=0.5
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.2),
            A.MotionBlur(p=0.1),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'], 
            clip=True, 
            min_area=1,
            min_visibility=0.1
            )
        )
    else:
        return A.Compose([
            ToTensorV2()
        ], 
        bbox_params=A.BboxParams(
            format='pascal_voc', 
            label_fields=['class_labels'],
            clip=True
            )
        )


# KITTI-MOTS -> Torchvision adapter
class KITTIMOTSToTorchvision(torch.utils.data.Dataset):
    """
    Returns:
      image: FloatTensor[C,H,W] in [0,1]
      target: dict(boxes, labels, image_id, area, iscrowd)
    """
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def raw_anns_to_target(self, raw_anns: List[Any], image_id: int) -> Dict[str, torch.Tensor]:
        boxes, labels = [], []

        for ann in raw_anns:
            cls = int(getattr(ann, "class_id", -1))
            box = getattr(ann, "bbox_xyxy", None)
            if box is None:
                continue

            boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
            labels.append(int(cls))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)

        area = (
            (boxes_t[:, 2] - boxes_t[:, 0]).clamp(min=0) *
            (boxes_t[:, 3] - boxes_t[:, 1]).clamp(min=0)
        ) if boxes_t.numel() else torch.zeros((0,), dtype=torch.float32)

        return {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area.to(torch.float32),
            "iscrowd": torch.zeros((labels_t.shape[0],), dtype=torch.int64)}

    def __getitem__(self, idx: int):
        img_pil, raw_anns, _ = self.base_ds[idx]
        img_np = np.array(img_pil)
        target = self.raw_anns_to_target(raw_anns, image_id=idx)
        return img_np, target


# Apply Transforms to each image
class ApplyAlbumentations(torch.utils.data.Dataset):
    def __init__(self, ds, tf, keep_empty: bool = True):
        self.ds = ds # already adapted
        self.tf = tf
        self.keep_empty = keep_empty
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img_np, target = self.ds[idx]

        boxes = target["boxes"].numpy().tolist()
        labels = target["labels"].numpy().tolist()

        if self.tf is not None:
            out = self.tf(image=img_np, bboxes=boxes, class_labels=labels)
            img = out["image"]
            boxes_tf = out["bboxes"]
            labels_tf = out["class_labels"]

            if isinstance(img, torch.Tensor):
                if img.dtype == torch.uint8:
                    img = img.float().div(255.0)
                else:
                    img = img.float()
            else:
                # safety fallback if transform didn't convert to tensor
                img = to_tensor(img)  # -> float [0,1]
        else:
            img = to_tensor(img_np)
            boxes_tf, labels_tf = boxes, labels

        # after albumentations the boxes could be removed...avoid them
        if len(boxes_tf) == 0 and not self.keep_empty:
            return self.__getitem__((idx + 1) % len(self))

        if len(boxes_tf) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        else:
            target["boxes"] = torch.tensor(boxes_tf, dtype=torch.float32)
            target["labels"] = torch.tensor(labels_tf, dtype=torch.int64)

        target["area"] = (
            (target["boxes"][:, 2] - target["boxes"][:, 0]).clamp(min=0) *
            (target["boxes"][:, 3] - target["boxes"][:, 1]).clamp(min=0)
        ) if target["boxes"].numel() else torch.zeros((0,), dtype=torch.float32)

        return img, target


@dataclass
class Detection:
    boxes: torch.Tensor   # (N,4) xyxy
    scores: torch.Tensor  # (N,)
    labels: torch.Tensor  # (N,)
    masks: Optional[torch.Tensor] = None  # (N,H,W) or None
    

# Faster R-CNN
class FasterRCNNModel(torch.nn.Module):
    def __init__(self, device: Optional[str] = None):
        super().__init__()
        self.device = device if device is not None else "cpu"
        self.model = self.build_model()

    def build_model(self) -> torch.nn.Module:
        return fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    
    def prepare_finetune(self, num_classes, train_backbone):
        # Replace box predictor head
        in_feats = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)

        # Freeze/unfreeze backbone
        for p in self.model.backbone.parameters():
            p.requires_grad = train_backbone

        self.model.to(self.device)
    
    def forward(self, images, targets=None):
        return self.model(images, targets)

    @torch.inference_mode()
    def predict(self, images: Sequence[Any], score_thresh: float = 0.5) -> List[Detection]:
        """
        images: list of torch tensors [C,H,W] in [0,1]
        returns: List[Detection]
        """
        self.model.eval()
        imgs = [img.to(self.device) for img in images]
        outputs = self.model(imgs)

        dets: List[Detection] = []
        for d in outputs:
            keep = d["scores"] >= score_thresh
            dets.append(
                Detection(
                    boxes=d["boxes"][keep].detach().cpu(),
                    scores=d["scores"][keep].detach().cpu(),
                    labels=d["labels"][keep].detach().cpu(),
                    masks=d.get("masks", None),
                )
            )
        return dets
        
        

