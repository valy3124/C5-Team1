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
import re  # kept for any future use; strip_thinking uses str.split instead
from PIL import Image as PILImage
from tqdm import tqdm

from dataset import VizWizDataset

DATA_BASE_DIR = '/ghome/group01/C5/vali/C5-Team1/Week3/dataset/VizWiz'
VAL_ANN       = os.path.join(DATA_BASE_DIR, 'annotations', 'val.json')
VAL_IMG_DIR   = os.path.join(DATA_BASE_DIR, 'images', 'val')
TRAIN_ANN     = os.path.join(DATA_BASE_DIR, 'annotations', 'train.json')
TRAIN_IMG_DIR = os.path.join(DATA_BASE_DIR, 'images', 'train')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Models that use chain-of-thought / thinking tokens
THINKING_MODELS = {'Qwen2.5-VL-7B-Instruct', 'Qwen3-VL-8B-Instruct', 'Qwen3.5-9B'}

# Thinking is disabled for all models, so a unified token budget is fine.
# Keeping a slightly larger budget for formerly-thinking models as a safety margin.
MAX_TOKENS_THINKING = 256
MAX_TOKENS_DEFAULT  = 128

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
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Maximum new tokens to generate. Defaults to 1024 for thinking models, 128 otherwise.")
    return parser.parse_args()


def load_model_and_processor(model_type):
    print(f"Loading Model: {model_type} loading onto {DEVICE} in bfloat16 to fit in memory...")

    if model_type == 'Qwen2-VL-7B-Instruct':
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


def strip_thinking(text):
    """
    Remove thinking blocks from model output.
    Handles:
      - Tagged complete:   <think>...</think>answer  → take everything after </think>
      - Tagged truncated:  <think>...                → take everything before <think>
      - No tags at all:    return as-is (Qwen2-VL, or thinking already disabled)
    Uses split() rather than regex for reliable boundary detection.
    Never returns an empty string if the original had content.
    """
    # Best case: clean split on closing tag — take everything after the last </think>
    if "</think>" in text:
        answer = text.split("</think>")[-1].strip()
        return answer if answer else text

    # Opening tag present but no closing tag — truncated inside thinking block,
    # take whatever was output before thinking started
    if "<think>" in text:
        before_think = text.split("<think>")[0].strip()
        return before_think if before_think else text

    # No tags — return as-is
    return text


def generate_caption(img, model, processor, model_type, max_tokens):
    prompt_text = (
        "Provide a brief and concise caption for this image. "
        "Answer in a single short sentence (less than 20 words)."
    )
    # Note: "do not explain your reasoning" removed — thinking models handle
    # CoT internally and the instruction could suppress useful reasoning.

    if model_type in ['Qwen2-VL-7B-Instruct', 'Qwen2.5-VL-7B-Instruct',
                      'Qwen3-VL-8B-Instruct', 'Qwen3.5-9B']:

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": prompt_text}
            ]}
        ]

        # ------------------------------------------------------------------ #
        # Disable thinking for all models:                                    #
        # - Qwen3.5: native enable_thinking=False flag                        #
        # - Qwen2.5-VL / Qwen3-VL: system prompt suppression + strip_thinking#
        #   as a safety net in case any <think> tokens still slip through.    #
        # - Qwen2-VL: no thinking capability, no action needed.               #
        # ------------------------------------------------------------------ #
        if model_type == 'Qwen3.5-9B':
            input_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # native flag — cleanest suppression
            )
        elif model_type in ('Qwen2.5-VL-7B-Instruct', 'Qwen3-VL-8B-Instruct'):
            messages_with_system = [
                {"role": "system", "content": "You are a helpful assistant. Do not output any thinking or reasoning. Reply directly with the answer only."},
                *messages
            ]
            input_text = processor.apply_chat_template(
                messages_with_system,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        inputs = processor(
            text=[input_text],
            images=[img],
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        out_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()

        # Strip any residual thinking blocks (applies mainly to Qwen2.5 / Qwen3-VL)
        out_text = strip_thinking(out_text)
        return out_text

    elif model_type == "llama3.2-11b":
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
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return generated_texts[0].strip()

    else:
        raise ValueError("Invalid model type")


def eval_dataset(model, processor, dataset, model_type, max_tokens, out_dir):
    all_preds     = []
    all_refs      = []
    all_img_ids   = []
    all_img_names = []

    for i in tqdm(range(len(dataset)), desc="Evaluating VLM"):
        img, img_id, img_name = dataset[i]

        pred = generate_caption(img, model, processor, model_type, max_tokens)
        all_preds.append(pred)

        img_id_val = img_id.item() if hasattr(img_id, "item") else img_id
        refs = dataset.image_captions[img_id_val]
        all_refs.append(refs)
        all_img_ids.append(img_id_val)
        all_img_names.append(img_name)

    # ------------------------------------------------------------------ #
    # Sanity-check: warn if many predictions are empty                    #
    # ------------------------------------------------------------------ #
    empty_count = sum(1 for p in all_preds if not p.strip())
    if empty_count > 0:
        print(f"WARNING: {empty_count}/{len(all_preds)} predictions are empty — "
              f"thinking stripping may still be incomplete.")

    # Compute metrics
    metrics = {}
    try:
        bleu1 = bleu.compute(predictions=all_preds, references=all_refs, max_order=1)
        bleu2 = bleu.compute(predictions=all_preds, references=all_refs, max_order=2)
        res_r = rouge.compute(predictions=all_preds, references=all_refs)
        res_m = meteor.compute(predictions=all_preds, references=all_refs)

        cider_score = 0.0
        if CIDER_AVAILABLE:
            cider_res   = cider.compute(predictions=all_preds, references=all_refs)
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

    # Resolve max_tokens: explicit arg > model-based default
    if args.max_tokens is not None:
        max_tokens = args.max_tokens
    elif args.model_type in THINKING_MODELS:
        max_tokens = MAX_TOKENS_THINKING
        print(f"Thinking model detected — using max_tokens={max_tokens}")
    else:
        max_tokens = MAX_TOKENS_DEFAULT

    val_split   = "val_search"   if args.mode == "search" else "val"
    val_ann     = TRAIN_ANN      if args.mode == "search" else VAL_ANN
    val_img_dir = TRAIN_IMG_DIR  if args.mode == "search" else VAL_IMG_DIR

    dataset_valid = VizWizDataset(
        annotation_file=val_ann,
        img_dir=val_img_dir,
        split=val_split,
        mode=args.mode,
        return_raw_img=True
    )

    model, processor = load_model_and_processor(args.model_type)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name  = f"eval_LLM_{args.model_type}_{args.mode}_{timestamp}"
    out_dir   = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    metrics, preds, refs, img_ids, img_names = eval_dataset(
        model, processor, dataset_valid, args.model_type, max_tokens, out_dir
    )

    # Qualitative visualisation
    sample_indices = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    viz_dir = os.path.join(out_dir, "visual_samples")
    os.makedirs(viz_dir, exist_ok=True)

    for idx in sample_indices:
        if idx >= len(preds):
            continue

        img_name  = img_names[idx]
        pred_str  = preds[idx]
        ref_strs  = refs[idx]

        try:
            img_path = os.path.join(val_img_dir, img_name)
            img      = PILImage.open(img_path).convert('RGB')

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

    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({**vars(args), "resolved_max_tokens": max_tokens}, f, indent=4)

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    history = [
        {"img_id": img_id, "prediction": pred, "references": ref}
        for pred, ref, img_id in zip(preds, refs, img_ids)
    ]
    with open(os.path.join(out_dir, 'predictions.json'), 'w') as f:
        json.dump(history, f, indent=4)

    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()