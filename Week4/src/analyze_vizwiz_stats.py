import json
import os
import numpy as np
from transformers import AutoTokenizer

DATA_BASE_DIR = '/ghome/group01/C5/vali/C5-Team1/Week3/dataset/VizWiz'
VAL_ANN       = os.path.join(DATA_BASE_DIR, 'annotations', 'val.json')
TRAIN_ANN     = os.path.join(DATA_BASE_DIR, 'annotations', 'train.json')

def analyze_file(filepath, tokenizer):
    print(f"Analyzing {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # annotations is a list of dicts with 'caption' key
    # Or in VizWiz, it might be images + annotations
    if 'annotations' in data:
        captions = [ann['caption'] for ann in data['annotations']]
    else:
        # Fallback if structure is different
        print("Expected 'annotations' key not found. Checking root...")
        captions = [item['caption'] for item in data if 'caption' in item]

    if not captions:
        print("No captions found!")
        return

    word_counts = [len(c.split()) for c in captions]
    
    # Tokenize in batches for speed
    token_counts = []
    batch_size = 1000
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i+batch_size]
        tokens = tokenizer(batch, add_special_tokens=False)['input_ids']
        token_counts.extend([len(t) for t in tokens])

    stats = {
        "count": len(captions),
        "words": {
            "min": int(np.min(word_counts)),
            "max": int(np.max(word_counts)),
            "avg": float(np.mean(word_counts)),
            "median": float(np.median(word_counts)),
            "p95": float(np.percentile(word_counts, 95))
        },
        "tokens": {
            "min": int(np.min(token_counts)),
            "max": int(np.max(token_counts)),
            "avg": float(np.mean(token_counts)),
            "median": float(np.median(token_counts)),
            "p95": float(np.percentile(token_counts, 95))
        }
    }
    return stats

def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    val_stats = analyze_file(VAL_ANN, tokenizer)
    train_stats = analyze_file(TRAIN_ANN, tokenizer)
    
    print("\n--- Validation Set Statistics ---")
    print(json.dumps(val_stats, indent=4))
    
    print("\n--- Training Set Statistics ---")
    print(json.dumps(train_stats, indent=4))

if __name__ == "__main__":
    main()
