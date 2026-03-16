import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate

from dataset import VizWizDataset, idx2char, char2idx
from model import ImageCaptioningModel

# --- Configuration ---
DATA_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'VizWiz')
TRAIN_ANN = os.path.join(DATA_BASE_DIR, 'annotations', 'train.json')
VAL_ANN = os.path.join(DATA_BASE_DIR, 'annotations', 'val.json')
TRAIN_IMG_DIR = os.path.join(DATA_BASE_DIR, 'images', 'train')
VAL_IMG_DIR = os.path.join(DATA_BASE_DIR, 'images', 'val')

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load HF evaluate metrics
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')

def convert_indices_to_string(indices):
    """
    Decodes a sequence of char indices back into a string, stopping at <EOS> or <PAD>.
    """
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
    print("Loading datasets...")
    dataset_train = VizWizDataset(TRAIN_ANN, TRAIN_IMG_DIR, split="train")
    dataset_valid = VizWizDataset(VAL_ANN, VAL_IMG_DIR, split="val")
    
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print("Initializing model...")
    model = ImageCaptioningModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit = nn.CrossEntropyLoss()
    
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, optimizer, crit, dataloader_train, epoch)
        print(f"End of Epoch {epoch} - Average Train Loss: {loss:.4f}")
        
        metrics = eval_epoch(model, dataloader_valid)
        print("Validation Metrics:")
        print(f"BLEU-1: {metrics.get('BLEU-1', 0):.2f}% | BLEU-2: {metrics.get('BLEU-2', 0):.2f}% | ROUGE-L: {metrics.get('ROUGE-L', 0):.2f}% | METEOR: {metrics.get('METEOR', 0):.2f}%")

if __name__ == "__main__":
    main()
