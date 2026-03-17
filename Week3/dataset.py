import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

# Vocabulary definition (baseline: character-level)
chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CHAR = len(chars)
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}

TEXT_MAX_LEN = 201

class VizWizDataset(Dataset):
    def __init__(self, annotation_file, img_dir, split="train", mode="search"):
        """
        Args:
            annotation_file (str): Path to the JSON annotation file.
            img_dir (str): Directory with all the images.
            split (str): 'train', 'val', or 'test'. Test set has no publicly available captions.
            mode (str): 'search' to use a subset of the dataset, 'full' to use the whole dataset.
        """
        self.img_dir = img_dir
        self.split = split
        self.mode = mode
        self.max_len = TEXT_MAX_LEN
        
        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
            
            if self.split == "train_search":
                num_samples = int(len(self.valid_image_ids) * 0.9)
                self.valid_image_ids = self.valid_image_ids[:num_samples]
                print(f"[{split} | {mode}] Using 90% subset: {len(self.valid_image_ids)} images.")
            elif self.split == "val_search":
                num_samples = int(len(self.valid_image_ids) * 0.9)
                self.valid_image_ids = self.valid_image_ids[num_samples:]
                print(f"[{split} | {mode}] Using 10% subset: {len(self.valid_image_ids)} images.")
            else:
                num_samples = int(len(self.valid_image_ids) * 0.1)
                self.valid_image_ids = self.valid_image_ids[:num_samples]
                print(f"[{split} | {mode}] Using generic 10% subset: {len(self.valid_image_ids)} images.")
        else:
            print(f"[{split} | {mode}] Using all {len(self.valid_image_ids)} images.")

        # Create samples list
        self.samples = []
        if self.split == 'test':
            for img_id in self.valid_image_ids:
                self.samples.append((img_id, None))
        elif 'val' in self.split or self.split == 'validation':
            # For validation, we use 1 sample per image (for loss) but keep access to all captions (for metrics)
            for img_id in self.valid_image_ids:
                self.samples.append((img_id, 0)) # Use first caption for loss
        else:
            # For training, treat each valid caption as a separate sample
            for img_id in self.valid_image_ids:
                for cap_idx in range(len(self.image_captions[img_id])):
                    self.samples.append((img_id, cap_idx))
        
        print(f"[{split}] Dataset initialized with {len(self.samples)} samples.")
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_id, cap_idx = self.samples[idx]
        img_name = self.images[img_id]
        
        # Load and process image
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = self.img_proc(img)
        
        if self.split == 'test':
            return img, img_id, img_name
            
        # Select the specific caption for this sample
        caption = self.image_captions[img_id][cap_idx]
        
        # Process caption to character sequence
        cap_list = list(caption)
        
        # Build sequence: <SOS> + chars + <EOS> + <PAD>...
        final_list = ['<SOS>']
        final_list.extend([c for c in cap_list if c in char2idx]) # Filter unknown characters just in case
        final_list.extend(['<EOS>'])
        
        gap = self.max_len - len(final_list)
        if gap > 0:
            final_list.extend(['<PAD>'] * gap)
        else:
            final_list = final_list[:self.max_len]
            final_list[-1] = '<EOS>'
            
        cap_idx = [char2idx[char] for char in final_list]
        
        return img, torch.tensor(cap_idx, dtype=torch.long), img_name
