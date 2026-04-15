import os
import sys
import argparse
import json
import torch
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BlipForConditionalGeneration, AutoProcessor
from pathlib import Path

# Add Week4 src to path for dataset reuse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Week4/src')))
try:
    from dataset import VizWizDataset
except ImportError:
    raise ImportError("Could not import VizWizDataset from Week4/src")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load HF evaluate metrics
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Clusters per sample")
    parser.add_argument("--model_path", type=str, 
                        default="/ghome/group01/C5/benet/C5-Team1/Week5/models/finetune_blip_both-finetune_full_20260324_220017/best_model",
                        help="Path to the trained model directory")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val"])
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input cluster assignments CSV")
    parser.add_argument("--data_dir", type=str, default="/ghome/group01/C5/vali/C5-Team1/Week3/dataset/VizWiz")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Loading Model from {args.model_path} onto {DEVICE}...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = BlipForConditionalGeneration.from_pretrained(args.model_path)
    model.to(DEVICE)
    model.eval()

    print(f"Loading CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    ann_file = os.path.join(args.data_dir, 'annotations', f'{args.split}.json')
    img_dir  = os.path.join(args.data_dir, 'images', args.split)

    # Note: mode='full' prevents the fixed search subset
    dataset = VizWizDataset(
        annotation_file=ann_file,
        img_dir=img_dir,
        split=args.split,
        mode="full",
        processor=processor
    )
    
    # Filter dataset strictly to images found in CSV
    filename_to_id = {v: k for k, v in dataset.images.items()}
    csv_filenames = set(df['filename'].tolist())
    valid_dataset_ids = set(dataset.valid_image_ids)
    
    filtered_samples = []
    for fname in df['filename']:
        if fname in filename_to_id:
            img_id = filename_to_id[fname]
            if img_id in valid_dataset_ids:
                filtered_samples.append((img_id, 0)) # cap_idx doesn't matter for evaluation
                
    dataset.samples = filtered_samples
    dataset.valid_image_ids = [s[0] for s in filtered_samples]
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    all_preds = []
    all_refs = []
    all_filenames = []

    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            pixel_values, img_ids, img_names = batch
            inputs = {"pixel_values": pixel_values.to(DEVICE)}
            
            # Predict
            out_ids = model.generate(**inputs, max_new_tokens=30)
            
            # Decode predictions
            preds = processor.batch_decode(out_ids, skip_special_tokens=True)
            preds = [p.strip() for p in preds]
            
            all_preds.extend(preds)
            all_filenames.extend(img_names)
            
            # Prepare references
            for i in range(len(img_ids)):
                img_id = img_ids[i].item() if hasattr(img_ids[i], "item") else img_ids[i]
                all_refs.append(dataset.image_captions[img_id])

    print("Computing metrics per sample...")
    results_dict = {}
    for i in tqdm(range(len(all_preds)), desc="Scoring"):
        pred = all_preds[i]
        refs = all_refs[i]
        fname = all_filenames[i]
        
        try:
            b1 = bleu.compute(predictions=[pred], references=[refs], max_order=1)
            b1_score = b1['bleu'] * 100 if b1 else 0.0
        except: b1_score = 0.0
        
        try:
            b2 = bleu.compute(predictions=[pred], references=[refs], max_order=2)
            b2_score = b2['bleu'] * 100 if b2 else 0.0
        except: b2_score = 0.0
            
        try:
            r = rouge.compute(predictions=[pred], references=[refs])
            r_score = r['rougeL'] * 100 if r else 0.0
        except: r_score = 0.0
            
        try:
            m = meteor.compute(predictions=[pred], references=[refs])
            m_score = m['meteor'] * 100 if m else 0.0
        except: m_score = 0.0
            
        results_dict[fname] = {
            'BLEU-1': b1_score,
            'BLEU-2': b2_score,
            'ROUGE-L': r_score,
            'METEOR': m_score,
            'PREDICTION': pred
        }

    # Enrich df
    df['BLEU-1'] = df['filename'].map(lambda x: results_dict.get(x, {}).get('BLEU-1', np.nan))
    df['BLEU-2'] = df['filename'].map(lambda x: results_dict.get(x, {}).get('BLEU-2', np.nan))
    df['ROUGE-L'] = df['filename'].map(lambda x: results_dict.get(x, {}).get('ROUGE-L', np.nan))
    df['METEOR'] = df['filename'].map(lambda x: results_dict.get(x, {}).get('METEOR', np.nan))
    df['PREDICTION'] = df['filename'].map(lambda x: results_dict.get(x, {}).get('PREDICTION', ""))

    # Save enriched samples CSV
    samples_out = args.csv_path.replace('.csv', '_with_metrics.csv')
    df.to_csv(samples_out, index=False)
    print(f"Saved enriched samples to {samples_out}")
    
    # Compute aggregate per cluster
    # We ignore NaN naturally with pandas
    cluster_stats = df.groupby('cluster')[['BLEU-1', 'BLEU-2', 'ROUGE-L', 'METEOR']].mean().reset_index()
    # Also add count of images per cluster
    counts = df.groupby('cluster').size().reset_index(name='count')
    cluster_stats = cluster_stats.merge(counts, on='cluster')
    
    # Sort descending by METEOR
    cluster_stats = cluster_stats.sort_values(by='METEOR', ascending=False)
    
    agg_out = args.csv_path.replace('.csv', '_cluster_averages.csv')
    cluster_stats.to_csv(agg_out, index=False)
    print(f"Saved cluster aggregations to {agg_out}")

if __name__ == "__main__":
    main()
