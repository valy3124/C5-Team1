import json
import evaluate
import warnings

warnings.filterwarnings('ignore')

cider = evaluate.load('sunhill/cider')
path = "/ghome/group01/C5/vali/C5-Team1/Week3/results/baseline/resnet18_lr0.0005_bs128_cmiqlnrw/captions_history.json"
with open(path, 'r') as f:
    data = json.load(f)

print("--- Retroactive CIDEr Scores for Baseline Model ---")
for ep in range(1, 13):
    records = data[str(ep)]
    if not records: continue
    
    preds = [r['prediction'] for r in records]
    refs = [r['references'] for r in records]
    
    score = cider.compute(predictions=preds, references=refs)['cider_score'] * 100
    print(f"Epoch {ep}: CIDEr = {score:.2f}")

