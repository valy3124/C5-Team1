import os
import json

from nltk.tokenize import word_tokenize
try:
    import evaluate
except ImportError:
    pass

import random

# Base paths
BASE_DIR = '/ghome/group01/C5/vali/C5-Team1/Week3/dataset/VizWiz'
VAL_ANN = os.path.join(BASE_DIR, 'annotations', 'val.json')

def load_data():
    with open(VAL_ANN, 'r') as f:
        data = json.load(f)
    
    # Map image id to captions
    img_to_caps = {}
    for ann in data['annotations']:
        if not ann.get('is_precanned', False) and not ann.get('is_rejected', False):
            img_id = ann['image_id']
            if img_id not in img_to_caps:
                img_to_caps[img_id] = []
            img_to_caps[img_id].append(ann['caption'])
            
    # Map image id to filename
    img_to_name = {img['id']: img['file_name'] for img in data['images']}
    
    return img_to_caps, img_to_name

def compute_metrics(prediction, references):
    import evaluate
    import warnings
    warnings.filterwarnings('ignore')
    
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    
    # We need lists of predictions and lists of lists of references
    preds = [prediction]
    refs = [references]
    
    b1 = bleu.compute(predictions=preds, references=refs, max_order=1)['bleu'] * 100
    b2 = bleu.compute(predictions=preds, references=refs, max_order=2)['bleu'] * 100
    ro = rouge.compute(predictions=preds, references=refs)['rougeL'] * 100
    me = meteor.compute(predictions=preds, references=refs)['meteor'] * 100
    
    return b1, b2, ro, me

def analyze_qualitative_examples():
    img_to_caps, img_to_name = load_data()
    
    # Find some interesting images (with >= 4 valid captions)
    valid_imgs = [img_id for img_id, caps in img_to_caps.items() if len(caps) >= 4]
    
    # Create artificial predictions to demonstrate metric behaviors
    examples = [
        {
            "desc": "Perfect Match (Exact words and order)",
            "img_id": valid_imgs[10],
            "refs": img_to_caps[valid_imgs[10]][:4],
            # We copy a reference exactly
            "pred": img_to_caps[valid_imgs[10]][0] 
        },
        {
            "desc": "Good Semantic Match (Different synonyms/structure)",
            "img_id": valid_imgs[20],
            "refs": ["A computer monitor displaying a blue error screen.", 
                     "A laptop screen showing a blue screen of death.",
                     "A blue computer screen with white writing on it.",
                     "A close up of a computer monitor with a blue error screen."],
            "pred": "A laptop monitor showing an error message on a blue background."
        },
        {
            "desc": "Partial Match (Only gets a few words right)",
            "img_id": valid_imgs[30],
            "refs": ["A bottle of medication with a white cap and yellow label",
                     "A close up of a bottle of pills",
                     "A small bottle of prescription medication.",
                     "A pill bottle with a white lid and yellow label in the center"],
            "pred": "A bottle of water with a white cap"
        },
        {
            "desc": "Poor Match (Hallucination/Wrong object)",
            "img_id": valid_imgs[40],
            "refs": ["A person holding a remote control for a television.",
                     "A hand holding a black remote control.",
                     "A TV remote control being held by a hand.",
                     "A close up of a person holding a black remote control in their hand."],
            "pred": "A close up of a cell phone on a table."
        }
    ]
    
    print("\n" + "="*80)
    print("QUALITATIVE EXAMPLES FOR METRICS (BLEU-1, BLEU-2, ROUGE-L, METEOR)")
    print("="*80 + "\n")
    
    for ex in examples:
        print(f"[{ex['desc']}]")
        # print(f"Image File: {img_to_name[ex['img_id']]}")
        print(f"References:")
        for i, ref in enumerate(ex['refs']):
            print(f"  {i+1}. {ref}")
        print(f"\nPrediction: \"{ex['pred']}\"")
        
        try:
            b1, b2, ro, me = compute_metrics(ex['pred'], ex['refs'])
            print(f"\nMetrics:")
            print(f"  BLEU-1 : {b1:6.2f}  (Measures exact 1-word overlap)")
            print(f"  BLEU-2 : {b2:6.2f}  (Measures exact 2-word phrase overlap)")
            print(f"  ROUGE-L: {ro:6.2f}  (Measures longest common sequence, order matters)")
            print(f"  METEOR : {me:6.2f}  (Allows stemming/synonyms, better for meaning)")
        except Exception as e:
            print(f"Error computing metrics: {e}")
            
        print("-" * 80 + "\n")

if __name__ == "__main__":
    analyze_qualitative_examples()
