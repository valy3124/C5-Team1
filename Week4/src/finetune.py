import os
import argparse
import time
import json
import torch
import evaluate
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
import random

from dataset import VizWizDataset

DATA_BASE_DIR = '/ghome/group01/C5/vali/C5-Team1/Week3/dataset/VizWiz'
TRAIN_ANN     = os.path.join(DATA_BASE_DIR, 'annotations', 'train.json')
TRAIN_IMG_DIR = os.path.join(DATA_BASE_DIR, 'images', 'train')
VAL_ANN       = os.path.join(DATA_BASE_DIR, 'annotations', 'val.json')
VAL_IMG_DIR   = os.path.join(DATA_BASE_DIR, 'images', 'val')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Metrics
bleu   = evaluate.load('bleu')
rouge  = evaluate.load('rouge')
meteor = evaluate.load('meteor')
try:
    cider = evaluate.load('sunhill/cider')
    CIDER_AVAILABLE = True
except Exception as e:
    print(f"Cider metric not available: {e}")
    CIDER_AVAILABLE = False


class TrainDatasetWrapper(Dataset):
    """Wraps the VizWizDataset to return tokenized text labels for training."""
    def __init__(self, base_dataset, tokenizer, max_length=32):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Unroll multiple captions so our length is larger
        self.train_samples = []
        for img_id in self.base_dataset.valid_image_ids:
            captions = self.base_dataset.image_captions[img_id]
            for cap in captions:
                self.train_samples.append((img_id, cap))
                
    def __len__(self):
        return len(self.train_samples)
        
    def __getitem__(self, idx):
        img_id, caption = self.train_samples[idx]
        img_name = self.base_dataset.images[img_id]
        
        # Load and process image using the fast transforms logic from base_dataset
        img_path = os.path.join(self.base_dataset.img_dir, img_name)
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        
        pixel_values = self.base_dataset.img_proc(img)
        
        # Tokenize caption
        text_inputs = self.tokenizer(
            caption, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        labels = text_inputs.input_ids.squeeze(0)
        # Replacing padding token id's with -100 so they are ignored by cross-entropy loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return pixel_values, labels

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune vit-bert on VizWiz")
    parser.add_argument("--model_type", type=str, default="blip", choices=["blip", "vit-gpt2", "vit-bert"])
    parser.add_argument("--strategy", type=int, choices=[1, 2, 3], required=True,
                        help="1: ViT finetune & BERT frozen | 2: ViT frozen & BERT finetune | 3: Both finetuned")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mode", type=str, default="search", choices=["full", "search"])
    parser.add_argument("--output_dir", type=str, default="../results")
    return parser.parse_args()


def eval_epoch(model, processor, dataloader):
    model.eval()
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values, img_ids, _ = batch
            inputs = {"pixel_values": pixel_values.to(DEVICE)}
            
            # Predict
            out_ids = model.generate(**inputs, max_new_tokens=30)
            
            # Decode predictions
            preds = processor.batch_decode(out_ids, skip_special_tokens=True)
            preds = [p.strip() for p in preds]
            
            all_preds.extend(preds)
            
            # Prepare references
            for i in range(len(img_ids)):
                img_id = img_ids[i].item() if hasattr(img_ids[i], "item") else img_ids[i]
                all_refs.append(dataloader.dataset.image_captions[img_id])
                
    try:
        bleu1 = bleu.compute(predictions=all_preds, references=all_refs, max_order=1)
        bleu2 = bleu.compute(predictions=all_preds, references=all_refs, max_order=2)
        res_r = rouge.compute(predictions=all_preds, references=all_refs)
        res_m = meteor.compute(predictions=all_preds, references=all_refs)

        cider_score = 0.0
        if CIDER_AVAILABLE:
            cider_res = cider.compute(predictions=all_preds, references=all_refs)
            cider_score = cider_res.get('cider_score', 0.0) * 100

        metrics = {
            "BLEU-1":  bleu1['bleu'] * 100 if bleu1 else 0.0,
            "BLEU-2":  bleu2['bleu'] * 100 if bleu2 else 0.0,
            "ROUGE-L": res_r['rougeL'] * 100 if res_r else 0.0,
            "METEOR":  res_m['meteor'] * 100 if res_m else 0.0,
            "CIDEr":   cider_score,
        }
    except Exception as e:
        print(f"Failed computing metrics: {e}")
        metrics = {k: 0 for k in ["BLEU-1", "BLEU-2", "ROUGE-L", "METEOR", "CIDEr"]}

    return metrics


def main():
    args = parse_args()
    
    # Initialize WandB
    wandb.init(project="C5-Week4-ImageCaptioning", config=vars(args))
    
    # Update args if WandB config is passing new ones (e.g. during sweeps)
    for key, value in wandb.config.items():
        setattr(args, key, value)
    
    print(f"Config: {vars(args)}")

    # Strategy Name
    strategy_map = {
        1: "vision-finetune_text-frozen",
        2: "vision-frozen_text-finetune",
        3: "both-finetune"
    }
    strategy_name = strategy_map[args.strategy]
    wandb.run.name = f"{strategy_name}_{args.mode}_{wandb.run.id}"

    print(f"Loading {args.model_type}...")
    from transformers import BlipForConditionalGeneration, VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer

    if args.model_type == "blip":
        model_name = "Salesforce/blip-image-captioning-base"
        img_processor = AutoProcessor.from_pretrained(model_name)
        tokenizer = img_processor.tokenizer
        model = BlipForConditionalGeneration.from_pretrained(model_name, use_safetensors=True)
        vision_params = model.vision_model.parameters()
        text_params = model.text_decoder.parameters()
        
    elif args.model_type == "vit-gpt2":
        model_name = "nlpconnect/vit-gpt2-image-captioning"
        img_processor = AutoImageProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name, use_safetensors=True)
        vision_params = model.encoder.parameters()
        text_params = model.decoder.parameters()
        
    elif args.model_type == "vit-bert":
        model_name = "atasoglu/vit-bert-flickr8k"
        img_processor = AutoImageProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        vision_params = model.encoder.parameters()
        text_params = model.decoder.parameters()

    # Configure Freezing Strategy
    model.train()
    
    if args.strategy == 1:
        print("Strategy 1: Freezing Text Decoder...")
        for param in text_params:
            param.requires_grad = False
        for param in vision_params:
            param.requires_grad = True
            
    elif args.strategy == 2:
        print("Strategy 2: Freezing Vision Encoder...")
        for param in text_params:
            param.requires_grad = True
        for param in vision_params:
            param.requires_grad = False
            
    elif args.strategy == 3:
        print("Strategy 3: Unfreezing everything...")
        for param in model.parameters():
            param.requires_grad = True

    model.to(DEVICE)

    train_split_name = "train_search" if args.mode == "search" else "train"
    val_split_name = "val_search" if args.mode == "search" else "val"

    # Datasets
    print("Loading datasets...")
    base_train = VizWizDataset(
        annotation_file=TRAIN_ANN,
        img_dir=TRAIN_IMG_DIR,
        split=train_split_name,
        mode=args.mode,
        processor=img_processor
    )
    dataset_train = TrainDatasetWrapper(base_train, tokenizer)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_ann = TRAIN_ANN if args.mode == "search" else VAL_ANN
    val_img_dir = TRAIN_IMG_DIR if args.mode == "search" else VAL_IMG_DIR
    dataset_valid = VizWizDataset(
        annotation_file=val_ann,
        img_dir=val_img_dir,
        split=val_split_name,
        mode=args.mode,
        processor=img_processor
    )
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Output Saving Setup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"finetune_{args.model_type}_{strategy_name}_{args.mode}_{timestamp}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Results will be saved to {out_dir}")
    
    best_meteor = -1.0
    best_epoch = -1
    best_metrics = {}

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader_train, desc="Training"):
            pixel_values, labels = batch
            pixel_values = pixel_values.to(DEVICE)
            labels = labels.to(DEVICE)
            
            if args.model_type == "blip":
                input_ids = labels.clone()
                input_ids[input_ids == -100] = tokenizer.pad_token_id
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
            else:
                outputs = model(pixel_values=pixel_values, labels=labels)
            
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            wandb.log({"train/step_loss": loss.item()})
            
        avg_train_loss = total_loss / len(dataloader_train)
        print(f"Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        print("Evaluating phase...")
        metrics = eval_epoch(model, tokenizer, dataloader_valid)
        metrics["train_loss"] = avg_train_loss
        print(f"Validation Metrics: {metrics}")
        
        # Logging to wandb
        wandb_logs = {
            "train/epoch_loss": avg_train_loss,
            "epoch": epoch
        }
        for k, v in metrics.items():
            if k != "train_loss":
                wandb_logs[f"val/{k}"] = v
        wandb.log(wandb_logs)
        
        # Track best METEOR
        if metrics["METEOR"] > best_meteor:
            best_meteor = metrics["METEOR"]
            best_epoch = epoch
            best_metrics = metrics
            
            # Save the best model
            save_dir = os.path.join("../results", f"best_model_{args.model_type}_mode_{args.mode}_strategy_{args.strategy}")
            print(f"New best METEOR! Saving model and processor to {save_dir}...")
            model.save_pretrained(save_dir)
            
            # For BLIP, img_processor exists; for TR, processor exists
            try:
                if args.model_type == "blip":
                    img_processor.save_pretrained(save_dir)
                else:
                    processor.save_pretrained(save_dir)
            except Exception as e:
                print(f"Could not save processor: {e}")
            
            # Save best model
            model.save_pretrained(os.path.join(out_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(out_dir, "best_model"))
            img_processor.save_pretrained(os.path.join(out_dir, "best_model"))
            
    print(f"\n=== Training Complete! ===")
    print(f"Best Epoch: {best_epoch} (Based on METEOR)")
    print(f"Best Metrics: {best_metrics}")
    
    # Save best metrics into summary so it shows elegantly in wandb dashboard
    wandb.run.summary["best_epoch"] = best_epoch
    for k, v in best_metrics.items():
        if k != "train_loss":
            wandb.run.summary[f"best_val/{k}"] = v

    wandb.finish()
        
if __name__ == "__main__":
    main()
