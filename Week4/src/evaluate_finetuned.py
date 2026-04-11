import os
import argparse
import time
import json
import torch
import evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, AutoProcessor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import textwrap
from pathlib import Path

from dataset import VizWizDataset

# Constants matching finetune.py
DATA_BASE_DIR = '/ghome/group01/C5/vali/C5-Team1/Week3/dataset/VizWiz'
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Finetuned BLIP Model on VizWiz")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the best_model directory")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mode", type=str, default="full", choices=["full", "search"])
    return parser.parse_args()


def load_model_and_processor(model_path):
    print(f"Loading Model from {model_path} onto {DEVICE}...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = BlipForConditionalGeneration.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    return model, processor


def eval_epoch(model, processor, dataloader, out_dir):
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
                
    # Compute metrics
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

    # Save visual samples matching evaluate_pretrained.py
    sample_indices = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    viz_dir = os.path.join(out_dir, "visual_samples")
    os.makedirs(viz_dir, exist_ok=True)
    
    samples_to_print = []
    for b in sample_indices:
        if b >= len(all_preds):
            continue
            
        img_id = list(dataloader.dataset.samples)[b][0]
        img_name = dataloader.dataset.images[img_id]
        pred_str = all_preds[b]
        ref_strs = all_refs[b]
        
        samples_to_print.append({
            "img_id": img_id,
            "img_name": img_name,
            "prediction": pred_str,
            "references": ref_strs
        })
        
        # Draw image and captions
        try:
            img_path = os.path.join(dataloader.dataset.img_dir, img_name)
            img = PILImage.open(img_path).convert('RGB')
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img)
            ax.axis('off')
            
            wrapped_pred = textwrap.fill(f"Pred: {pred_str}", width=60)
            wrapped_refs = textwrap.fill(f"Ref: {ref_strs[0]}", width=60)
            if len(ref_strs) > 1:
                wrapped_refs += "\n" + textwrap.fill(f"Ref2: {ref_strs[1]}", width=60)
                
            plt.suptitle(wrapped_pred + "\n" + wrapped_refs, fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{img_name}_sample_{b}.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Could not print visualization for sample {b}: {e}")

    return metrics, all_preds, all_refs


def main():
    args = parse_args()
    print(f"Config: {vars(args)}")

    model, processor = load_model_and_processor(args.model_path)

    # Dataset setup
    dataset_valid = VizWizDataset(
        annotation_file=VAL_ANN,
        img_dir=VAL_IMG_DIR,
        split="val",
        mode=args.mode,
        processor=processor
    )
    
    dataloader_valid = DataLoader(
        dataset_valid, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Output Setup (Subfolder in model's parent directory)
    model_dir = Path(args.model_path).parent
    out_dir = model_dir / "qualitative_evaluation"
    os.makedirs(out_dir, exist_ok=True)
    
    # Run evaluation
    metrics, preds, refs = eval_epoch(model, processor, dataloader_valid, str(out_dir))
    
    print("\n================ Metrics ================")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("=========================================\n")

    # Save artifacts
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    history = []
    for i, (pred, ref) in enumerate(zip(preds, refs)):
        img_id = list(dataset_valid.samples)[i][0]
        history.append({
            "img_id": img_id,
            "prediction": pred,
            "references": ref
        })
    with open(out_dir / 'predictions.json', 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
