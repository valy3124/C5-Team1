import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

class VizWizDataset(Dataset):
    # Standard normalization presets matching Hugging Face defaults
    HF_MEAN = (0.5, 0.5, 0.5)
    HF_STD  = (0.5, 0.5, 0.5)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, annotation_file, img_dir, split="train", mode="search", processor=None, return_raw_img=False):
        """
        Args:
            annotation_file (str): Path to the JSON annotation file.
            img_dir (str): Directory with all the images.
            split (str): 'train', 'val', or 'test'.
            mode (str): 'search' (90/10 subset) or 'full' (entire split).
            processor: (Deprecated) Hugging Face processor to process the PIL image.
            return_raw_img: if True, returns the PIL Image instead of tensor (useful for LLMs).
        """
        self.img_dir = img_dir
        self.split   = split
        self.mode    = mode
        self.return_raw_img = return_raw_img
        
        # Decide transforms based on processor type or fallback to default
        if processor is not None and hasattr(processor, "image_mean"):
            mean = processor.image_mean
            std = processor.image_std
        else:
            mean = self.CLIP_MEAN
            std = self.CLIP_STD
            
        # Fast PyTorch transform pipeline identical to Week 3
        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize(mean, std),
        )

        # Load annotations
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            
        self.images = {img['id']: img['file_name'] for img in data['images']}
        
        if split != 'test':
            self.image_captions = {} # mapping from image_id -> list of valid captions
            for ann in data['annotations']:
                # Challenge rules: exclude precanned and spam captions
                if ann.get('is_precanned', False) or ann.get('is_rejected', False):
                    continue
                    
                img_id = ann['image_id']
                if img_id not in self.image_captions:
                    self.image_captions[img_id] = []
                self.image_captions[img_id].append(ann['caption'])
                
            # Keep only images that have at least one valid caption
            self.valid_image_ids = sorted(list(self.image_captions.keys()))
        else:
            self.valid_image_ids = sorted(list(self.images.keys()))
            
        if self.mode == "search":
            random.seed(42)
            # Shuffle the IDs once with fixed seed so splits are consistent
            random.shuffle(self.valid_image_ids)
            
            # Take only 10% of the dataset TOTAL for the search sweep to be fast!
            total_search_samples = int(len(self.valid_image_ids) * 0.1)
            search_ids = self.valid_image_ids[:total_search_samples]
            
            if "train" in self.split: # Use 80% of our small search subset
                num_samples = int(len(search_ids) * 0.8)
                self.valid_image_ids = search_ids[:num_samples]
                print(f"[{split} | {mode}] Using tiny train subset: {len(self.valid_image_ids)} images.")
            elif "val" in self.split: # Use 20% of our small search subset
                num_samples = int(len(search_ids) * 0.8)
                self.valid_image_ids = search_ids[num_samples:]
                print(f"[{split} | {mode}] Using tiny val subset: {len(self.valid_image_ids)} images.")
            else:
                self.valid_image_ids = search_ids
                print(f"[{split} | {mode}] Using generic search subset: {len(self.valid_image_ids)} images.")
        else:
            print(f"[{split} | {mode}] Using all {len(self.valid_image_ids)} images.")
            
        # Create samples list
        self.samples = []
        if self.split == 'test':
            for img_id in self.valid_image_ids:
                self.samples.append((img_id, None))
        else:
            # For validation/evaluation, we just need 1 sample per image (to generate 1 caption)
            # Since we look up references via ID later.
            for img_id in self.valid_image_ids:
                self.samples.append((img_id, 0)) # We'll just generate based on the image
        
        print(f"[{split}] Dataset initialized with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_id, cap_idx = self.samples[idx]
        img_name = self.images[img_id]
        
        # Load and process image using fast PyTorch transforms
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        if self.return_raw_img:
            if self.split == 'test':
                return img, img_id, img_name
            return img, img_id, img_name

        # Apply C++ accelerated transforms instead of slow HuggingFace processor
        pixel_values = self.img_proc(img)
        
        if self.split == 'test':
            return pixel_values, img_id, img_name
            
        # Return pixel_values and the image ID (or name) so we can fetch all references for evaluation
        return pixel_values, img_id, img_name
