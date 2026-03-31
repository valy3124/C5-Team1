import os
import argparse
import time
import wandb
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BlipVisionModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import evaluate

from dataset import VizWizDataset

# Data paths
DATA_BASE_DIR = '/ghome/group01/C5/vali/C5-Team1/Week3/dataset/VizWiz'
TRAIN_ANN     = os.path.join(DATA_BASE_DIR, 'annotations', 'train.json')
TRAIN_IMG_DIR = os.path.join(DATA_BASE_DIR, 'images', 'train')
VAL_ANN       = os.path.join(DATA_BASE_DIR, 'annotations', 'val.json')
VAL_IMG_DIR   = os.path.join(DATA_BASE_DIR, 'images', 'val')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Metrics
meteor = evaluate.load('meteor')

class CustomVLM(nn.Module):
    def __init__(self, vision_model_name, llm_name, lora_r=16):
        super().__init__()
        # Load Frozen Vision Encoder
        # Usually from the previously finetuned BLIP folder
        print(f"Loading Vision Encoder from {vision_model_name}...")
        self.vision_encoder = BlipVisionModel.from_pretrained(vision_model_name, use_safetensors=True).to(DEVICE)
        # Freeze Vision Encoder completely
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        print(f"Loading LLM {llm_name}...")
        if "Qwen2.5-VL" in llm_name:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                llm_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
        elif "Qwen3-VL" in llm_name:
            from transformers import Qwen3VLForConditionalGeneration
            self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
                llm_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
        elif "Qwen2-VL" in llm_name:
            from transformers import Qwen2VLForConditionalGeneration
            self.llm = Qwen2VLForConditionalGeneration.from_pretrained(
                llm_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name, 
                torch_dtype=torch.bfloat16, 
                device_map="auto" # Using auto will handle GPU mapping
            )
        
        # Apply LoRA to LLM
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()
        
        # Linear Projector: connects Vision -> LLM
        vision_hidden_size = self.vision_encoder.config.hidden_size
        llm_hidden_size = self.llm.config.hidden_size
        self.projector = nn.Linear(vision_hidden_size, llm_hidden_size, dtype=torch.bfloat16).to(DEVICE)
        
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        # 1. Extract visual features
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values.to(self.vision_encoder.device))
            # Shape: (batch_size, num_patches, hidden_size) e.g., (B, 197, 768)
            image_features = vision_outputs.last_hidden_state
            
        # 2. Project to LLM's embedding space
        image_embeds = self.projector(image_features.to(self.projector.weight.device).to(self.projector.weight.dtype))
        
        # 3. Get text embeddings from LLM
        # Ensure input_ids don't go to device 0 blindly if model is split
        inputs_embeds_txt = self.llm.get_input_embeddings()(input_ids.to(self.llm.device))
        
        # 4. Concatenate
        # Inputs: [Image_Embeds, Text_Embeds]
        inputs_embeds = torch.cat([image_embeds, inputs_embeds_txt], dim=1)
        
        # Expand attention mask for the prepended image tokens
        if attention_mask is not None:
            batch_size, seq_len = input_ids.shape
            num_image_tokens = image_embeds.shape[1]
            image_mask = torch.ones((batch_size, num_image_tokens), device=attention_mask.device, dtype=attention_mask.dtype)
            extended_attention_mask = torch.cat([image_mask, attention_mask], dim=1)
        else:
            extended_attention_mask = None
            
        # Expand labels: image tokens are shifted to -100 so they aren't predicted
        if labels is not None:
            batch_size = labels.shape[0]
            num_image_tokens = image_embeds.shape[1]
            image_labels = torch.full((batch_size, num_image_tokens), -100, device=labels.device, dtype=labels.dtype)
            extended_labels = torch.cat([image_labels, labels], dim=1)
        else:
            extended_labels = None
            
        # 5. Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels
        )
        return outputs

    def generate(self, pixel_values, tokenizer, max_new_tokens=30):
        # Very manual basic generation loop since standard `.generate` expects text inputs
        self.eval()
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values.to(self.vision_encoder.device))
            image_embeds = self.projector(vision_outputs.last_hidden_state.to(self.projector.weight.device).to(self.projector.weight.dtype))
            
            # Start token
            input_ids = torch.tensor([[tokenizer.bos_token_id] if tokenizer.bos_token_id else [tokenizer.eos_token_id]]).to(self.llm.device)
            
            for _ in range(max_new_tokens):
                txt_embeds = self.llm.get_input_embeddings()(input_ids)
                inputs_embeds = torch.cat([image_embeds, txt_embeds], dim=1)
                
                outputs = self.llm(inputs_embeds=inputs_embeds)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
        return input_ids[:, 1:]

class TrainDatasetWrapper(Dataset):
    def __init__(self, base_dataset, tokenizer, max_length=128):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_text = "Provide a brief and concise caption for this image. Answer in a single short sentence (less than 20 words). "
        
        self.train_samples = []
        for img_id in self.base_dataset.valid_image_ids:
            captions = self.base_dataset.image_captions[img_id]
            for cap in captions:
                self.train_samples.append((img_id, cap))
                
    def __len__(self):
        return len(self.train_samples)
        
    def __getitem__(self, idx):
        img_id, caption = self.train_samples[idx]
        
        # Manually load the image just like the dataset does
        img_name = self.base_dataset.images[img_id]
        img_path = os.path.join(self.base_dataset.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        pixel_values = self.base_dataset.img_proc(img)
        
        full_text = self.prompt_text + caption
        text_inputs = self.tokenizer(
            full_text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        input_ids = text_inputs.input_ids.squeeze()
        attention_mask = text_inputs.attention_mask.squeeze()
        
        # Replace padding with -100 for CE loss
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Mask out the prompt tokens from the loss
        prompt_len = len(self.tokenizer(self.prompt_text).input_ids)
        labels[:prompt_len] = -100
        
        return pixel_values, input_ids, attention_mask, labels

def eval_epoch(model, tokenizer, dataloader_valid, epoch, out_dir):
    model.eval()
    all_preds = []
    all_refs = []
    
    base_ds = dataloader_valid.dataset.base_dataset
    prompt_text = "Provide a brief and concise caption for this image. Answer in a single short sentence (less than 20 words). "
    
    sample_indices = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    valid_ids_sub = base_ds.valid_image_ids[:100].copy()
    for idx_val in sample_indices:
        if idx_val < len(base_ds.valid_image_ids) and base_ds.valid_image_ids[idx_val] not in valid_ids_sub:
            valid_ids_sub.append(base_ds.valid_image_ids[idx_val])
            
    viz_dir = os.path.join(out_dir, f"visual_samples_epoch_{epoch}")
    os.makedirs(viz_dir, exist_ok=True)
    
    for img_id in tqdm(valid_ids_sub, desc="Evaluating"):
        img_name = base_ds.images[img_id]
        img_path = os.path.join(base_ds.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        pixel_values = base_ds.img_proc(img)
        pixel_values = pixel_values.unsqueeze(0) # Batch size 1
        
        preds_ids = model.generate(pixel_values, tokenizer, max_new_tokens=128, prompt_text=prompt_text)
        pred_text = tokenizer.decode(preds_ids[0], skip_special_tokens=True)
        all_preds.append(pred_text)
        
        refs = base_ds.image_captions[img_id]
        all_refs.append(refs)
        
        original_idx = base_ds.valid_image_ids.index(img_id)
        if original_idx in sample_indices:
            try:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(img)
                ax.axis('off')
                
                wrapped_pred = textwrap.fill(f"Pred: {pred_text}", width=60)
                wrapped_refs = textwrap.fill(f"Ref: {refs[0]}", width=60)
                if len(refs) > 1:
                    wrapped_refs += "\n" + textwrap.fill(f"Ref2: {refs[1]}", width=60)
                    
                plt.suptitle(wrapped_pred + "\n" + wrapped_refs, fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"{img_name}_epoch_{epoch}.png"))
                plt.close(fig)
            except Exception as e:
                print(f"Visualization failed for sample {original_idx}: {e}")
                
    res_m = meteor.compute(predictions=all_preds, references=all_refs)
    return {"METEOR": res_m['meteor'] * 100 if res_m else 0.0}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4) # Higher LR for projector
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    # This should be your saved BLIP model path, or default Huggingface BLIP if not trained
    parser.add_argument("--vision_model", type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument("--mode", type=str, default="search", choices=["full", "search"])
    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(project="C5-Week4-CustomLoRA", config=vars(args))
    
    print("Loading Processors...")
    # Get image processor matching the vision encoder
    img_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # Get text tokenizer matching the LLM
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    # Qwen doesn't set pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading Dataset...")
    val_split = "val_search" if args.mode == "search" else "val"
    val_ann = TRAIN_ANN if args.mode == "search" else VAL_ANN
    val_img_dir = TRAIN_IMG_DIR if args.mode == "search" else VAL_IMG_DIR
    
    dataset_train = VizWizDataset(TRAIN_ANN, TRAIN_IMG_DIR, split='train_search' if args.mode=='search' else 'train', processor=img_processor)
    dataset_valid = VizWizDataset(val_ann, val_img_dir, split=val_split, processor=img_processor, mode=args.mode)
    
    wrapper_train = TrainDatasetWrapper(dataset_train, tokenizer)
    wrapper_valid = TrainDatasetWrapper(dataset_valid, tokenizer)
    
    dataloader_train = DataLoader(wrapper_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataloader_valid = DataLoader(wrapper_valid, batch_size=1, shuffle=False)
    
    print("Loading Model...")
    model = CustomVLM(vision_model_name=args.vision_model, llm_name=args.llm_model)
    
    from transformers import get_cosine_schedule_with_warmup
    # Optimizer only on projector and lora params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    accumulation_steps = args.gradient_accumulation
    num_training_steps = (len(dataloader_train) // accumulation_steps) * args.epochs
    num_warmup_steps = int(num_training_steps * 0.05)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    best_meteor = 0.0
    clean_model_name = args.llm_model.replace("/", "_")
    out_dir_base = os.path.join(args.output_dir, f"run_custom_vlm_{clean_model_name}_{args.mode}_{int(time.time())}")
    os.makedirs(out_dir_base, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(dataloader_train, desc="Training")):
            pixel_values, input_ids, attention_mask, labels = batch
            
            outputs = model(
                pixel_values=pixel_values, 
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader_train):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            total_loss += (loss.item() * accumulation_steps)
            wandb.log({"train/step_loss": loss.item() * accumulation_steps, "train/lr": lr_scheduler.get_last_lr()[0]})
            
        avg_train_loss = total_loss / len(dataloader_train)
        print(f"Train Loss: {avg_train_loss:.4f}")
        
        metrics = eval_epoch(model, tokenizer, dataloader_valid, epoch, out_dir_base)
        print(f"Validation Metrics: {metrics}")
        wandb.log({"train/epoch_loss": avg_train_loss, "val/METEOR": metrics["METEOR"], "epoch": epoch})
        
        if metrics["METEOR"] > best_meteor:
            best_meteor = metrics["METEOR"]
            save_dir = os.path.join(args.output_dir, f"best_custom_vlm_{clean_model_name}_{args.mode}_epoch_{epoch}")
            
            print(f"New best METEOR: {best_meteor}! Saving to {save_dir}...")
            os.makedirs(save_dir, exist_ok=True)
            # Save the trained specific parts manually
            model.llm.save_pretrained(os.path.join(save_dir, "lora_weights"))
            torch.save(model.projector.state_dict(), os.path.join(save_dir, "projector.pt"))
            tokenizer.save_pretrained(save_dir)
            
            with open(os.path.join(out_dir_base, "best_model_info.txt"), "w") as f:
                f.write(f"Best Model Epoch: {epoch}\nMETEOR: {best_meteor}\nPath: {save_dir}\n")
                
    wandb.finish()

if __name__ == "__main__":
    main()
