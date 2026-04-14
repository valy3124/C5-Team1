import os
import argparse
import time
import json
import torch
import evaluate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap
from PIL import Image as PILImage
from tqdm import tqdm

from dataset import VizWizDataset

DATA_BASE_DIR = '/ghome/group01/C5/vali/C5-Team1/Week3/dataset/VizWiz'
VAL_ANN       = os.path.join(DATA_BASE_DIR, 'annotations', 'val.json')
VAL_IMG_DIR   = os.path.join(DATA_BASE_DIR, 'images', 'val')
TRAIN_ANN     = os.path.join(DATA_BASE_DIR, 'annotations', 'train.json')
TRAIN_IMG_DIR = os.path.join(DATA_BASE_DIR, 'images', 'train')

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
    parser = argparse.ArgumentParser(description="Evaluate Large Multimodal Models (Llama 3.2, Qwen2-VL)")
    parser.add_argument("--model_type", type=str, default="Qwen3.5-9B", 
                        choices=["Qwen3.5-9B", "Qwen3-VL-8B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2-VL-7B-Instruct"],
                        help="Which Vision LLM to evaluate")
    parser.add_argument("--mode", type=str, default="search",
                        choices=["full", "search"],
                        help="'search' uses smaller subset of data, 'full' uses all.")
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument("--max_tokens", type=int, default=128,
                        help="Maximum new tokens to generate")
    return parser.parse_args()


def load_model_and_processor(model_type):
    print(f"Loading Model: {model_type} loading onto {DEVICE} in bfloat16 to fit in memory...")
    
    if model_type == 'Qwen2-VL-7B-Instruct':
        # Using a Qwen2-VL model equivalent roughly in scale
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        model_id = "Qwen/Qwen2-VL-7B-Instruct"
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    elif model_type == 'Qwen3-VL-8B-Instruct':
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    elif model_type == 'Qwen2.5-VL-7B-Instruct':
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    elif model_type == 'Qwen3.5-9B':
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model_id = "Qwen/Qwen3.5-9B"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        raise ValueError("Unknown model type")

    model.eval()
    return model, processor


def generate_caption(img, model, processor, model_type, max_tokens):
    prompt_text = "Provide a brief and concise caption for this image. Answer in a single short sentence (less than 20 words)."

    if model_type == "llama3.2-11b":
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=img, text=input_text, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
            
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return generated_texts[0].strip()
        
    elif model_type in ['Qwen2-VL-7B-Instruct', 'Qwen3-VL-8B-Instruct', 'Qwen3.5-9B', 'Qwen2.5-VL-7B-Instruct']:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[input_text], images=[img], padding=True, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        full_text = processor.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)[0]
        
        if "<|im_start|>assistant\n" in full_text:
            out_text = full_text.split("<|im_start|>assistant\n")[-1]
        elif "<|im_start|> assistant\n" in full_text:
            out_text = full_text.split("<|im_start|> assistant\n")[-1]
        elif "assistant\n" in full_text:
            out_text = full_text.split("assistant\n")[-1]
        else:
            generated_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            out_text = generated_texts[0]
            
        out_text = out_text.replace("<|im_end|>", "").strip()
        
        # If output contains a thinking block, strip it out completely
        import re
        if "<think>" in out_text:
            out_text = re.sub(r'<think>.*?</think>', '', out_text, flags=re.DOTALL)
            # if the generated text was cut off and </think> is missing:
            if "<think>" in out_text:
                out_text = re.sub(r'<think>.*', '', out_text, flags=re.DOTALL)
                
        return out_text.strip()
        
    else:
        raise ValueError("Invalid model type")


def eval_dataset(model, processor, dataset, model_type, max_tokens, out_dir):
    all_preds = []
    all_refs = []
    all_img_ids = []
    all_img_names = []
    
    # We iterate sequentially with index because batched inference with mixed image sizes in Vision LLMs is complex
    for i in tqdm(range(len(dataset)), desc="Evaluating VLM"):
        img, img_id, img_name = dataset[i]
        
        pred = generate_caption(img, model, processor, model_type, max_tokens)
        all_preds.append(pred)
        
        # Collect references
        img_id_val = img_id.item() if hasattr(img_id, "item") else img_id
        refs = dataset.image_captions[img_id_val]
        all_refs.append(refs)
        all_img_ids.append(img_id_val)
        all_img_names.append(img_name)

    # Compute metrics
    metrics = {}
    try:
        # Bleu expects a list of list of strings for references or tokenized.
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

    return metrics, all_preds, all_refs, all_img_ids, all_img_names

def main():
    args = parse_args()
    print(f"Config: {vars(args)}")

    val_split = "val_search" if args.mode == "search" else "val"
    val_ann = TRAIN_ANN if args.mode == "search" else VAL_ANN
    val_img_dir = TRAIN_IMG_DIR if args.mode == "search" else VAL_IMG_DIR
    
    # Notice we use return_raw_img=True because VLMs natively process the PIL image through apply_chat_template 
    dataset_valid = VizWizDataset(
        annotation_file=val_ann,
        img_dir=val_img_dir,
        split=val_split,
        mode=args.mode,
        return_raw_img=True
    )
    
    model, processor = load_model_and_processor(args.model_type)
    
    # Output Saving Setup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"eval_LLM_{args.model_type}_{args.mode}_{timestamp}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Run loop
    metrics, preds, refs, img_ids, img_names = eval_dataset(model, processor, dataset_valid, args.model_type, args.max_tokens, out_dir)
    
    # Qualitative Logging
    sample_indices = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    viz_dir = os.path.join(out_dir, "visual_samples")
    os.makedirs(viz_dir, exist_ok=True)
    
    for idx in sample_indices:
        if idx >= len(preds):
            continue
        
        img_name = img_names[idx]
        img_id = img_ids[idx]
        pred_str = preds[idx]
        ref_strs = refs[idx]
        
        try:
            # Re-load image for visualization
            img_path = os.path.join(val_img_dir, img_name)
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
            plt.savefig(os.path.join(viz_dir, f"{img_name}_sample_{idx}.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Visualization failed for sample {idx}: {e}")
    
    print("\n================ Metrics ================")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("=========================================\n")

    # Save artifacts
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    history = []
    for i, (pred, ref, img_id) in enumerate(zip(preds, refs, img_ids)):
        history.append({
            "img_id": img_id,
            "prediction": pred,
            "references": ref
        })
    with open(os.path.join(out_dir, 'predictions.json'), 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
