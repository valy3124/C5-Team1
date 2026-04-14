import os
import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoTokenizer
from safetensors.torch import load_file

from dataset import VizWizDataset
from train_custom_vlm import CustomVLM

DATA_BASE_DIR = '/ghome/group01/C5/vali/C5-Team1/Week3/dataset/VizWiz'
VAL_ANN       = os.path.join(DATA_BASE_DIR, 'annotations', 'val.json')
VAL_IMG_DIR   = os.path.join(DATA_BASE_DIR, 'images', 'val')
TRAIN_ANN     = os.path.join(DATA_BASE_DIR, 'annotations', 'train.json')
TRAIN_IMG_DIR = os.path.join(DATA_BASE_DIR, 'images', 'train')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to saved model folder (e.g. MODELS_ARRANGED/results/best_custom_vlm_search_Qwen_Qwen3.5-0.8B)")
    parser.add_argument("--vision_model", type=str, default="/ghome/group01/C5/benet/C5-Team1/Week4/MODELS_ARRANGED/Task1/Task1.2/finetune_blip_both-finetune_full_20260324_220017/best_model")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--mode", type=str, default="search", choices=["full", "search"])
    parser.add_argument("--output_dir", type=str, default="visualizations")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("Loading image processor...")
    img_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    print("Loading dataset...")
    val_split   = "val_search"   if args.mode == "search" else "val"
    val_ann     = TRAIN_ANN      if args.mode == "search" else VAL_ANN
    val_img_dir = TRAIN_IMG_DIR  if args.mode == "search" else VAL_IMG_DIR

    dataset_valid = VizWizDataset(
        annotation_file=val_ann,
        img_dir=val_img_dir,
        split=val_split,
        mode=args.mode,
        processor=img_processor
    )

    print(f"Initializing CustomVLM with {args.llm_model} ...")
    model = CustomVLM(vision_model_name=args.vision_model, llm_name=args.llm_model)

    print(f"Loading weights from {args.model_path} ...")
    # Load projector
    model.projector.load_state_dict(torch.load(os.path.join(args.model_path, "projector.pt"), map_location=DEVICE))
    
    # Load LoRA
    lora_weights_path = os.path.join(args.model_path, "lora_weights", "adapter_model.safetensors")
    if os.path.exists(lora_weights_path):
        state_dict = load_file(lora_weights_path)
        model.llm.load_state_dict(state_dict, strict=False)
    else:
        print("WARNING: LoRA weights not found at", lora_weights_path)

    model.eval()
    
    out_dir = os.path.join(args.output_dir, os.path.basename(os.path.normpath(args.model_path)))
    os.makedirs(out_dir, exist_ok=True)
    
    # Visualization
    sample_indices = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    print(f"Generating visualizations in {out_dir} ...")
    
    for idx in sample_indices:
        if idx >= len(dataset_valid.valid_image_ids):
            continue
            
        img_id = dataset_valid.valid_image_ids[idx]
        img_name = dataset_valid.images[img_id]
        img_path = os.path.join(dataset_valid.img_dir, img_name)
        
        try:
            img = PILImage.open(img_path).convert('RGB')
            # Extra processing for generation
            pixel_values = dataset_valid.img_proc(img).unsqueeze(0).to(DEVICE)
            
            pred_str = model.generate(pixel_values, tokenizer, max_new_tokens=30)
            ref_strs = dataset_valid.image_captions[img_id]
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img)
            ax.axis('off')

            wrapped_pred = textwrap.fill(f"Pred [{args.llm_model}]: {pred_str}", width=60)
            wrapped_refs = textwrap.fill(f"Ref: {ref_strs[0]}", width=60)
            if len(ref_strs) > 1:
                wrapped_refs += "\n" + textwrap.fill(f"Ref2: {ref_strs[1]}", width=60)

            plt.suptitle(wrapped_pred + "\n" + wrapped_refs, fontsize=12)
            plt.tight_layout()
            save_path = os.path.join(out_dir, f"{img_name}_sample_{idx}.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved {save_path}")
        except Exception as e:
            print(f"Failed on index {idx}: {e}")

    # Test on a completely black/blank image
    print("Testing on a blank/black image...")
    try:
        blank_img = PILImage.new('RGB', (224, 224), color='black')
        pixel_values = dataset_valid.img_proc(blank_img).unsqueeze(0).to(DEVICE)
        
        pred_str = model.generate(pixel_values, tokenizer, max_new_tokens=30)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(blank_img)
        ax.axis('off')

        wrapped_pred = textwrap.fill(f"Pred [{args.llm_model}]: {pred_str}", width=60)
        wrapped_refs = "Ref: [Blank/Black Image - No Reference]"

        plt.suptitle(wrapped_pred + "\n" + wrapped_refs, fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(out_dir, "blank_black_image_sample.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved {save_path}")
    except Exception as e:
        print(f"Failed on blank image: {e}")

if __name__ == "__main__":
    main()
