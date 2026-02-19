"""
Wrapper so it is easier to load the annotations and the images from the KITTI MOST dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, Literal, Any, Optional

import numpy as np
import PIL.Image as Image
import pycocotools.mask as rletools



@dataclass(frozen=True, slots=True)
class InstanceAnn:
    object_id: int
    class_id: int
    instance_id: int
    mask_rle: Dict[str, Any]            
    bbox_xyxy: Tuple[int, int, int, int] # (x1, y1, x2, y2)


class KITTIMOTS:
    """
    Kitti Mots dataset wrapper.

    __getitem__ returns:
      (PIL.Image RGB, list[InstanceAnn])
    """
    # COCO:  1=Person, 3=Car
    # KITTI: 1=Car, 2=Pedestrian
    LABELS_MAPPING = {
        1: 3,  # KITTI Car → COCO Car
        2: 1   # KITTI Pedestrian → COCO Person
    }
    IGNORE_ID = 10000
    BG_ID = 0
    VALIDATION_SEQS = {2, 6, 7, 8, 10, 13, 14, 16, 18}

    def __init__(
        self,
        root: Union[str, Path] = "~/mcv/datasets/C5/KITTI-MOTS/",
        split: Literal["train", "validation"] = "train",
        ann_source: Literal["png", "txt"] = "txt",
        id_divisor: int = 1000,
        compute_boxes: bool = True,
    ):
        
        self.root = Path(root).expanduser()
        self.split = split

        self.ann_source = ann_source
        self.id_divisor = int(id_divisor)
        self.compute_boxes = bool(compute_boxes)

        self.img_root = self.root / "training" / "image_02"
        self.png_ann_root = self.root / "instances"
        self.txt_ann_root = self.root / "instances_txt"

        if ann_source == "png" and not self.png_ann_root.exists():
            raise FileNotFoundError(f"PNG annotations folder not found: {self.png_ann_root}")
        if ann_source == "txt" and not self.txt_ann_root.exists():
            raise FileNotFoundError(f"TXT annotations folder not found: {self.txt_ann_root}")

        # Build (seq, frame, img_path)
        self.index: List[Tuple[str, int, Path]] = self._build_index()

        # Cache parsed txt per sequence
        self._txt_cache: Dict[str, Dict[int, List[InstanceAnn]]] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Tuple[Image.Image, List[InstanceAnn]]:
        seq, frame, img_path = self.index[i]
        image = Image.open(img_path).convert("RGB")

        if self.ann_source == "png":
            anns = self._anns_from_png(seq, frame)
        else:
            anns = self._anns_from_txt(seq, frame)

        meta = {
            "seq": seq,
            "frame": frame,
            "image_path": str(img_path),
            "index": i,
        }

        return image, anns, meta


    def _build_index(self) -> List[Tuple[str, int, Path]]:
        index = []
        seq_dirs = sorted([p for p in self.img_root.iterdir() if p.is_dir()])

        for seq_dir in seq_dirs:
            seq = seq_dir.name              
            seq_int = int(seq)           

            is_val_seq = (seq_int in self.VALIDATION_SEQS)

            if self.split == "train":
                if is_val_seq:
                    continue
            
            elif self.split == "validation":
                if not is_val_seq:
                    continue

            else:
                raise ValueError(f"Unknown split: {self.split}")

            img_paths = sorted(seq_dir.glob("*.png"))   
            for img_path in img_paths:
                frame = int(img_path.stem)
                index.append((seq, frame, img_path))

        return index


    # PNG annotations: instances/<seq>/<frame>.png
    def _anns_from_png(self, seq: str, frame: int) -> List[InstanceAnn]:
        ann_path = self.png_ann_root / seq / f"{frame:06d}.png"
        if not ann_path.exists():
            return []

        ann = np.array(Image.open(ann_path))  # uint16 ids
        obj_ids = np.unique(ann)

        mask = np.zeros(ann.shape, dtype=np.uint8, order="F")
        out: List[InstanceAnn] = []

        for obj_id in obj_ids:
            obj_id = int(obj_id)
            if obj_id in (self.BG_ID, self.IGNORE_ID):
                continue

            class_id = obj_id // self.id_divisor
            instance_id = obj_id % self.id_divisor
            if class_id not in self.LABELS_MAPPING:
                continue

            mask.fill(0)
            mask[ann == obj_id] = 1
            rle = rletools.encode(mask)

            if self.compute_boxes:
                bbox = self._bbox_from_binary(mask)
            else:
                bbox = (0, 0, 0, 0)

            out.append(InstanceAnn(obj_id, class_id, instance_id, rle, bbox))

        return out

    # TXT annotations: instances_txt/<seq>.txt
    def _anns_from_txt(self, seq: str, frame: int) -> List[InstanceAnn]:
        if seq not in self._txt_cache:
            txt_path = self.txt_ann_root / f"{seq}.txt"
            if not txt_path.exists():
                raise FileNotFoundError(f"TXT annotations not found: {txt_path}")
            self._txt_cache[seq] = self._parse_txt(txt_path)

        return self._txt_cache[seq].get(frame, [])

    def _parse_txt(self, txt_path: Path) -> Dict[int, List[InstanceAnn]]:
        per_frame: Dict[int, List[InstanceAnn]] = {}

        with txt_path.open("r") as f:
            for line in f:
                fields = line.strip().split(" ")
                if len(fields) < 6:
                    continue

                t = int(fields[0])
                obj_id = int(fields[1])
                class_id = int(fields[2])
                h, w = int(fields[3]), int(fields[4])
                counts = fields[5].encode("UTF-8")

                if obj_id == self.IGNORE_ID:
                    continue
                if class_id not in self.LABELS_MAPPING:
                    continue

                rle = {"size": [h, w], "counts": counts}
                instance_id = obj_id % self.id_divisor

                if self.compute_boxes:
                    bbox = self._bbox_from_rle(rle)
                else:
                    bbox = (0, 0, 0, 0)

                per_frame.setdefault(t, []).append(InstanceAnn(obj_id, class_id, instance_id, rle, bbox))

        return per_frame

    # BBox helpers
    @staticmethod
    def _bbox_from_binary(mask: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    @staticmethod
    def _bbox_from_rle(rle: Dict[str, Any]) -> Tuple[int, int, int, int]:
        m = rletools.decode(rle)
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


class DEArt:
    """
    DEArt dataset wrapper.

    __getitem__ returns:
      (PIL.Image RGB, list[InstanceAnn])
    """
    
# TODO: IMPLEMENT THE DEART DATASET READER + SPLIT WITH FIXED SEED + UTILS AS IN KITTI-MOTS

if __name__ == "__main__":
    import random
    from PIL import ImageDraw

    # The main dataset root (in this case, in the UAB cluster)
    ROOT = "~/mcv/datasets/C5/KITTI-MOTS/"

    # Instantiate the class, will be fed with the png masks and will compute bboxes
    ds = KITTIMOTS(
        root=ROOT,
        split="train",
        ann_source="png",
        compute_boxes=True,
    )

    # Print some of the data it provides
    print(ds)
    print("Dataset length:", len(ds))

    i = random.randint(0, len(ds) - 1)
    img, anns, meta = ds[i]

    print("\nSample index:", i)
    print("Seq:", meta["seq"], "Frame:", meta["frame"])
    print("Image path:", meta["image_path"])
    print("Image size (W,H):", img.size)
    print("Num instances:", len(anns))

    # Print first 5 instances
    for k, a in enumerate(anns[:5]):
        print(
            f"  inst[{k}] object_id={a.object_id} class_id={a.class_id} instance_id={a.instance_id} bbox={a.bbox_xyxy}"
        )

    vis = img.copy()
    draw = ImageDraw.Draw(vis)

    overlay = np.array(vis).copy()
    vis_overlay = vis

    # Draw bboxes and overlay the masks
    if len(anns) > 0:
        for a in anns:
            draw.rectangle(a.bbox_xyxy, outline=(0, 255, 0), width=2)

            mask = rletools.decode(a.mask_rle).astype(np.uint8)

            color = np.array([
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            ])

            overlay[mask == 1] = color

        vis_overlay = Image.fromarray(
            (0.6 * np.array(vis) + 0.4 * overlay).astype(np.uint8)
        )   

        
    out_path = Path("kitti_mots_debug_sample.jpg")
    vis_overlay.save(out_path)
    print(f"Saved visualization to: {out_path.resolve()}")