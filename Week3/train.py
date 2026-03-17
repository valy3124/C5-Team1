import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import wandb

from dataset import VizWizDataset, idx2char, char2idx
from model import ImageCaptioningModel, ENCODER_CONFIGS

# --- Paths ---
DATA_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'VizWiz')
TRAIN_ANN = os.path.join(DATA_BASE_DIR, 'annotations', 'train.json')
VAL_ANN = os.path.join(DATA_BASE_DIR, 'annotations', 'val.json')
TRAIN_IMG_DIR = os.path.join(DATA_BASE_DIR, 'images', 'train')
VAL_IMG_DIR = os.path.join(DATA_BASE_DIR, 'images', 'val')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load HF evaluate metrics
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')


def parse_args():
    parser = argparse.ArgumentParser(description='Train image captioning model on VizWiz')
    parser.add_argument('--encoder',     type=str,   default='resnet18', choices=list(ENCODER_CONFIGS),
                        help='Encoder backbone to use')
    parser.add_argument('--epochs',      type=int,   default=5)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--output_dir',  type=str,   default='results/checkpoints',
                        help='Directory to save best model checkpoint')
    parser.add_argument('--project',     type=str,   default='C5-ImageCaptioning',
                        help='W&B project name')
    return parser.parse_args()

def convert_indices_to_string(indices):
    res = ""
    for idx in indices:
        char = idx2char[idx.item()]
        if char == '<EOS>' or char == '<PAD>':
            break
        if char != '<SOS>':
            res += char
    return res

def train_one_epoch(model, optimizer, crit, dataloader, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    for img, caption in progress_bar:
        img, caption = img.to(DEVICE), caption.to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(img)
        loss = crit(pred, caption)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader):
    model.eval()
    all_preds = []
    all_refs = []
    
    print("Evaluating...")
    with torch.no_grad():
        for img, caption in tqdm(dataloader, desc="Eval"):
            img = img.to(DEVICE)
            # caption is shape (batch_size, seq_len)
            
            # shape of pred: (batch_size, NUM_CHAR, seq_len)
            pred = model(img)
            
            # Extract argmax to get character indices: (batch_size, seq_len)
            pred_indices = pred.argmax(dim=1)
            
            # Decode strings line by line
            for b in range(img.size(0)):
                pred_str = convert_indices_to_string(pred_indices[b])
                ref_str = convert_indices_to_string(caption[b])
                
                all_preds.append(pred_str)
                # HF evaluate expects a list of list format for references if calculating corpus-level metrics
                all_refs.append([ref_str]) 
                
    # Calculate metrics at the end of epoch
    # Note: If generated text is empty or too short, BLEU might crash or severely penalize.
    try:
        bleu1 = bleu.compute(predictions=all_preds, references=all_refs, max_order=1)
        bleu2 = bleu.compute(predictions=all_preds, references=all_refs, max_order=2)
        res_r = rouge.compute(predictions=all_preds, references=all_refs)
        res_m = meteor.compute(predictions=all_preds, references=all_refs)
        
        metrics = {
            "BLEU-1": bleu1['bleu'] * 100,
            "BLEU-2": bleu2['bleu'] * 100,
            "ROUGE-L": res_r['rougeL'] * 100,
            "METEOR": res_m['meteor'] * 100
        }
    except Exception as e:
        print(f"Failed computing metrics (possibly empty predictions): {e}")
        metrics = {"BLEU-1": 0, "BLEU-2": 0, "ROUGE-L": 0, "METEOR": 0}
        
    return metrics

def main():
    args = parse_args()

    run = wandb.init(
        project=args.project,
        config=vars(args),
        name=f"{args.encoder}_lr{args.lr}_bs{args.batch_size}",
    )
    cfg = wandb.config  # sweep may override args values

    print(f"Config: encoder={cfg.encoder}, lr={cfg.lr}, batch_size={cfg.batch_size}, epochs={cfg.epochs}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.output_dir, f"{run.name}.pt")

    print("Loading datasets...")
    dataset_train = VizWizDataset(TRAIN_ANN, TRAIN_IMG_DIR, split="train")
    dataset_valid = VizWizDataset(VAL_ANN, VAL_IMG_DIR, split="val")

    dataloader_train = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers)

    print(f"Initializing model with encoder: {cfg.encoder} ...")
    model = ImageCaptioningModel(encoder_name=cfg.encoder).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()

    best_bleu1 = -1.0

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, optimizer, crit, dataloader_train, epoch)
        print(f"End of Epoch {epoch} - Average Train Loss: {loss:.4f}")

        metrics = eval_epoch(model, dataloader_valid)
        print("Validation Metrics:")
        print(f"BLEU-1: {metrics.get('BLEU-1', 0):.2f}% | BLEU-2: {metrics.get('BLEU-2', 0):.2f}% | "
              f"ROUGE-L: {metrics.get('ROUGE-L', 0):.2f}% | METEOR: {metrics.get('METEOR', 0):.2f}%")

        wandb.log({"epoch": epoch, "train_loss": loss, **metrics})

        if metrics.get('BLEU-1', 0) > best_bleu1:
            best_bleu1 = metrics['BLEU-1']
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best BLEU-1: {best_bleu1:.2f}% — checkpoint saved to {ckpt_path}")
            wandb.run.summary["best_BLEU-1"] = best_bleu1

    wandb.finish()


if __name__ == "__main__":
    main()
