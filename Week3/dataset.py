import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

# Vocabulary definition (baseline: character-level) - Kept for backward compatibility
chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CHAR = len(chars)
idx2char = {k: v for k, v in enumerate(chars)}
char2idx = {v: k for k, v in enumerate(chars)}

TEXT_MAX_LEN = 201

class BaseTokenizer:
    def __init__(self, max_len=TEXT_MAX_LEN):
        self.max_len = max_len
        self.vocab_size = 0
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.sos_idx = 0
        self.eos_idx = 0
        self.pad_idx = 0

    def encode(self, text):
        raise NotImplementedError

    def decode(self, indices):
        raise NotImplementedError

class CharTokenizer(BaseTokenizer):
    def __init__(self, max_len=TEXT_MAX_LEN):
        super().__init__(max_len)
        self.chars = chars
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.vocab_size = len(self.chars)
        self.sos_idx = self.char2idx[self.sos_token]
        self.eos_idx = self.char2idx[self.eos_token]
        self.pad_idx = self.char2idx[self.pad_token]

    def encode(self, text):
        cap_list = list(text)
        final_list = [self.sos_token]
        final_list.extend([c for c in cap_list if c in self.char2idx])
        final_list.append(self.eos_token)
        
        gap = self.max_len - len(final_list)
        if gap > 0:
            final_list.extend([self.pad_token] * gap)
        else:
            final_list = final_list[:self.max_len]
            final_list[-1] = self.eos_token
            
        return torch.tensor([self.char2idx[c] for c in final_list], dtype=torch.long)

    def decode(self, indices):
        res = ""
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            char = self.idx2char[idx]
            if char == self.eos_token or char == self.pad_token:
                break
            if char != self.sos_token:
                res += char
        return res

class SubwordTokenizer(BaseTokenizer):
    def __init__(self, model_id="openai/clip-vit-base-patch32", max_len=77):
        # CLIP uses a max_len of 77 by default
        super().__init__(max_len)
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.vocab_size = self.tokenizer.vocab_size
        self.sos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.pad_token = self.tokenizer.pad_token
        self.sos_idx = self.tokenizer.bos_token_id
        self.eos_idx = self.tokenizer.eos_token_id
        self.pad_idx = self.tokenizer.pad_token_id

    def encode(self, text):
        out = self.tokenizer(
            text, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )
        return out['input_ids'].squeeze(0)

    def decode(self, indices):
        # Filter out SOS/EOS/PAD for clean output if needed, or use tokenizer.decode
        return self.tokenizer.decode(indices, skip_special_tokens=True).strip()

class WordTokenizer(BaseTokenizer):
    def __init__(self, vocab=None, max_len=40):
        # 40 words is usually plenty for image captions
        super().__init__(max_len)
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        self.vocab = vocab or [self.sos_token, self.eos_token, self.pad_token, "<UNK>"]
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.sos_idx = self.word2idx[self.sos_token]
        self.eos_idx = self.word2idx[self.eos_token]
        self.pad_idx = self.word2idx[self.pad_token]
        self.unk_idx = self.word2idx["<UNK>"]

    def encode(self, text):
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text.lower())
        final_list = [self.sos_token]
        final_list.extend(tokens)
        final_list.append(self.eos_token)
        
        gap = self.max_len - len(final_list)
        if gap > 0:
            final_list.extend([self.pad_token] * gap)
        else:
            final_list = final_list[:self.max_len]
            final_list[-1] = self.eos_token
            
        indices = [self.word2idx.get(w, self.unk_idx) for w in final_list]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        res = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            word = self.idx2word[idx]
            if word == self.eos_token or word == self.pad_token:
                break
            if word != self.sos_token:
                res.append(word)
        return " ".join(res)

class VizWizDataset(Dataset):
    # Standard normalization presets
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    CLIP_MEAN     = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, annotation_file, img_dir, split="train", mode="search",
                 img_mean=None, img_std=None, tokenizer=None):
        """
        Args:
            annotation_file (str): Path to the JSON annotation file.
            img_dir (str): Directory with all the images.
            split (str): 'train', 'val', or 'test'.
            mode (str): 'search' (90/10 subset) or 'full' (entire split).
            img_mean (tuple): Normalization mean. Defaults to ImageNet values.
            img_std  (tuple): Normalization std.  Defaults to ImageNet values.
            tokenizer (BaseTokenizer): Tokenizer instance.
        """
        self.img_dir = img_dir
        self.split   = split
        self.mode    = mode
        self.tokenizer = tokenizer if tokenizer is not None else CharTokenizer()
        self.max_len = self.tokenizer.max_len

        mean = img_mean if img_mean is not None else self.IMAGENET_MEAN
        std  = img_std  if img_std  is not None else self.IMAGENET_STD

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
        
        # Process caption using the tokenizer
        cap_idx = self.tokenizer.encode(caption)
        
        return img, cap_idx, img_name

