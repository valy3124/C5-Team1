import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

import warnings
warnings.filterwarnings('ignore')

refs = [
    "a white paper showing an image of black and brown dog",
    "A library book with pictures of two dogs on the cover on a wooden table.",
    "A book with a black and a tan dog walking down a snowy street.",
    "The book cover shows two dogs in the snow",
    "A book cover title Dog Years with an image of a black and brown dog walking up the street, on the left side it has a due date sticker from a library."
]

predictions = {
    "1. Perfect Match": "The book cover shows two dogs in the snow",
    "2. Good Semantic Match": "A volume with hounds walking outdoors in winter weather",
    "3. Partial Match": "A book with a dog",
    "4. Poor Match": "A white car parked on a snowy street"
}

# Tokenize references
tokenized_refs = [word_tokenize(r.lower()) for r in refs]

print("\n=== QUALITATIVE METRIC EXAMPLES ===")
print("Image: VizWiz_val_00000002.jpg")
print("\nGround Truths:")
for i, r in enumerate(refs): print(f"  - {r}")

print("\n--- Predictions vs Metrics ---")
smooth = SmoothingFunction().method1

for scenario, pred in predictions.items():
    print(f"\nScenario: {scenario}")
    print(f"Prediction: \"{pred}\"")
    
    tokenized_pred = word_tokenize(pred.lower())
    
    # NLTK expects a list of tokenized references and one tokenized prediction
    b1 = sentence_bleu(tokenized_refs, tokenized_pred, weights=(1, 0, 0, 0), smoothing_function=smooth)
    b2 = sentence_bleu(tokenized_refs, tokenized_pred, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    
    # METEOR needs un-tokenized refs but it parses them. Wait, NLTK meteor_score expects lists of tokens for refs since nltk 3.6
    m = meteor_score(tokenized_refs, tokenized_pred)
    
    print(f"BLEU-1: {b1:.4f} | BLEU-2: {b2:.4f} | METEOR: {m:.4f}")
