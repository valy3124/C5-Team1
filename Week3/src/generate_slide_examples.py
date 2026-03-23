import evaluate
import json
import warnings

warnings.filterwarnings('ignore')

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

# Selected Ground Truth Captions (VizWiz_val_00000002.jpg)
refs = [
    "a white paper showing an image of black and brown dog",
    "A library book with pictures of two dogs on the cover on a wooden table.",
    "A book with a black and a tan dog walking down a snowy street.",
    "The book cover shows two dogs in the snow",
    "A book cover title Dog Years with an image of a black and brown dog walking up the street, on the left side it has a due date sticker from a library."
]

predictions = {
    "1. Perfect Match (Exact copy of GT)": "The book cover shows two dogs in the snow",
    "2. Good Semantic Match (Synonyms/Close meaning)": "A volume with hounds walking outdoors in winter weather",
    "3. Partial/Short Match (Safe prediction)": "A book with a dog",
    "4. Poor Match (Hallucination)": "A white car parked on a snowy street"
}

print("\n=== QUALITATIVE METRIC EXAMPLES ===")
print("Image File: VizWiz_val_00000002.jpg")
print("\nGround Truth Annotations:")
for i, ref in enumerate(refs):
    print(f"  {i+1}. {ref}")
    
print("\n--- Model Predictions vs Metrics ---")
for scenario, pred in predictions.items():
    print(f"\nScenario: {scenario}")
    print(f"Prediction: \"{pred}\"")
    
    # HF Evaluate expects list styles:
    b1 = bleu.compute(predictions=[pred], references=[refs], max_order=1)['bleu']
    b2 = bleu.compute(predictions=[pred], references=[refs], max_order=2)['bleu']
    m = meteor.compute(predictions=[pred], references=[refs])['meteor']
    
    print(f"> BLEU-1: {b1:.4f} | BLEU-2: {b2:.4f} | METEOR: {m:.4f}")
