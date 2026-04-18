import os
import json
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class VizWizDataset(Dataset):
    HF_MEAN   = (0.5, 0.5, 0.5)
    HF_STD    = (0.5, 0.5, 0.5)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, annotation_file, img_dir, split="train", mode="full",
                 processor=None, return_raw_img=False):
        self.img_dir       = img_dir
        self.split         = split
        self.mode          = mode
        self.return_raw_img = return_raw_img

        if processor is not None and hasattr(processor, "image_mean"):
            mean, std = processor.image_mean, processor.image_std
        else:
            mean, std = self.CLIP_MEAN, self.CLIP_STD

        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize(mean, std),
        )

        with open(annotation_file, "r") as f:
            data = json.load(f)

        self.images = {img["id"]: img["file_name"] for img in data["images"]}

        if split != "test":
            self.image_captions = {}
            for ann in data["annotations"]:
                if ann.get("is_precanned", False) or ann.get("is_rejected", False):
                    continue
                img_id = ann["image_id"]
                if img_id not in self.image_captions:
                    self.image_captions[img_id] = []
                self.image_captions[img_id].append(ann["caption"])
            self.valid_image_ids = sorted(list(self.image_captions.keys()))
        else:
            self.valid_image_ids = sorted(list(self.images.keys()))

        if self.mode == "search":
            random.seed(42)
            random.shuffle(self.valid_image_ids)
            total = int(len(self.valid_image_ids) * 0.1)
            search_ids = self.valid_image_ids[:total]
            if "train" in self.split:
                n = int(len(search_ids) * 0.8)
                self.valid_image_ids = search_ids[:n]
            elif "val" in self.split:
                n = int(len(search_ids) * 0.8)
                self.valid_image_ids = search_ids[n:]
            else:
                self.valid_image_ids = search_ids

        self.samples = [(img_id, 0) for img_id in self.valid_image_ids]
        print(f"[{split} | {mode}] Dataset: {len(self.samples)} images from '{img_dir}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, _ = self.samples[idx]
        img_name  = self.images[img_id]
        img_path  = os.path.join(self.img_dir, img_name)
        img       = Image.open(img_path).convert("RGB")

        if self.return_raw_img:
            return img, img_id, img_name

        pixel_values = self.img_proc(img)
        return pixel_values, img_id, img_name
