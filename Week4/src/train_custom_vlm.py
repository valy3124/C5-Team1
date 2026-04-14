import os
import argparse
import time
import wandb
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BlipVisionModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
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


def is_qwen35(model_name: str) -> bool:
    return "Qwen3.5" in model_name


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        answer = text.split("</think>")[-1].strip()
        return answer if answer else text
    if "<think>" in text:
        before_think = text.split("<think>")[0].strip()
        return before_think if before_think else text
    return text


class CustomVLM(nn.Module):
    def __init__(self, vision_model_name, llm_name, lora_r=16):
        super().__init__()
        self.llm_name = llm_name

        # --- Frozen Vision Encoder ---
        print(f"Loading Vision Encoder from {vision_model_name}...")
        self.vision_encoder = BlipVisionModel.from_pretrained(
            vision_model_name, use_safetensors=True
        ).to(DEVICE)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # --- LLM ---
        print(f"Loading LLM {llm_name}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Apply LoRA — frozen during Stage 1, unfrozen in Stage 2
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

        # --- MLP Projector (LLaVA-style, replaces single Linear) ---
        vision_hidden_size = self.vision_encoder.config.hidden_size
        llm_hidden_size    = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size, dtype=torch.bfloat16)
        ).to(DEVICE)

    # ---------------------------------------------------------------------- #
    # Stage helpers: freeze / unfreeze LoRA params                           #
    # ---------------------------------------------------------------------- #
    def freeze_llm(self):
        """Stage 1: train projector only."""
        for param in self.llm.parameters():
            param.requires_grad = False

    def unfreeze_llm_lora(self):
        """Stage 2: re-enable LoRA params for joint training."""
        for name, param in self.llm.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        with torch.no_grad():
            vision_outputs = self.vision_encoder(
                pixel_values=pixel_values.to(self.vision_encoder.device)
            )
            image_features = vision_outputs.last_hidden_state

        image_embeds = self.projector(
            image_features.to(self.projector[0].weight.device).to(self.projector[0].weight.dtype)
        )

        inputs_embeds_txt = self.llm.get_input_embeddings()(input_ids.to(self.llm.device))
        inputs_embeds     = torch.cat([image_embeds, inputs_embeds_txt], dim=1)

        if attention_mask is not None:
            attention_mask   = attention_mask.to(self.llm.device)
            num_image_tokens = image_embeds.shape[1]
            image_mask = torch.ones(
                (input_ids.shape[0], num_image_tokens),
                device=self.llm.device, dtype=attention_mask.dtype
            )
            extended_attention_mask = torch.cat([image_mask, attention_mask], dim=1)
        else:
            extended_attention_mask = None

        if labels is not None:
            labels           = labels.to(self.llm.device)
            num_image_tokens = image_embeds.shape[1]
            image_labels = torch.full(
                (labels.shape[0], num_image_tokens), -100,
                device=self.llm.device, dtype=labels.dtype
            )
            extended_labels = torch.cat([image_labels, labels], dim=1)
        else:
            extended_labels = None

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels
        )

    def generate(self, pixel_values, tokenizer, max_new_tokens=30):
        self.eval()
        with torch.no_grad():
            vision_outputs = self.vision_encoder(
                pixel_values=pixel_values.to(self.vision_encoder.device)
            )
            image_embeds = self.projector(
                vision_outputs.last_hidden_state
                .to(self.projector[0].weight.device)
                .to(self.projector[0].weight.dtype)
            )

            if is_qwen35(self.llm_name):
                messages = [{"role": "user", "content": "Describe this image briefly."}]
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompt_ids    = tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.llm.device)
                prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
                current_embeds = torch.cat([image_embeds, prompt_embeds], dim=1)
                generated_ids  = prompt_ids.clone()

                for _ in range(max_new_tokens):
                    out        = self.llm(inputs_embeds=current_embeds)
                    next_token = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    next_embed     = self.llm.get_input_embeddings()(next_token)
                    current_embeds = torch.cat([current_embeds, next_embed], dim=1)

                new_token_ids = generated_ids[:, prompt_ids.shape[1]:]
                decoded = tokenizer.decode(new_token_ids[0], skip_special_tokens=True).strip()
                return strip_thinking(decoded)

            else:
                bos_id    = tokenizer.bos_token_id or tokenizer.eos_token_id
                input_ids = torch.tensor([[bos_id]]).to(self.llm.device)

                for _ in range(max_new_tokens):
                    txt_embeds    = self.llm.get_input_embeddings()(input_ids)
                    inputs_embeds = torch.cat([image_embeds, txt_embeds], dim=1)
                    out           = self.llm(inputs_embeds=inputs_embeds)
                    next_token    = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
                    input_ids     = torch.cat([input_ids, next_token], dim=1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break

                return tokenizer.decode(input_ids[0, 1:], skip_special_tokens=True)


class TrainDatasetWrapper(Dataset):
    def __init__(self, base_dataset, tokenizer, max_length=64):
        self.base_dataset = base_dataset
        self.tokenizer    = tokenizer
        self.max_length   = max_length

        self.train_samples = []
        for img_id in self.base_dataset.valid_image_ids:
            for cap in self.base_dataset.image_captions[img_id]:
                self.train_samples.append((img_id, cap))

    def __len__(self):
        return len(self.train_samples)

    def __getitem__(self, idx):
        img_id, caption = self.train_samples[idx]

        img_name = self.base_dataset.images[img_id]
        img_path = os.path.join(self.base_dataset.img_dir, img_name)
        img      = Image.open(img_path).convert('RGB')
        pixel_values = self.base_dataset.img_proc(img)

        text_inputs = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids      = text_inputs.input_ids.squeeze()
        attention_mask = text_inputs.attention_mask.squeeze()
        labels         = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return pixel_values, input_ids, attention_mask, labels


def eval_epoch(model, tokenizer, dataloader_valid):
    model.eval()
    all_preds, all_refs = [], []

    base_ds       = dataloader_valid.dataset.base_dataset
    valid_ids_sub = base_ds.valid_image_ids[:100]

    for img_id in tqdm(valid_ids_sub, desc="Evaluating"):
        img_name = base_ds.images[img_id]
        img_path = os.path.join(base_ds.img_dir, img_name)
        img      = Image.open(img_path).convert('RGB')
        pixel_values = base_ds.img_proc(img).unsqueeze(0)

        pred_text = model.generate(pixel_values, tokenizer)
        all_preds.append(pred_text)
        all_refs.append(base_ds.image_captions[img_id])

    res_m = meteor.compute(predictions=all_preds, references=all_refs)
    return {"METEOR": res_m['meteor'] * 100 if res_m else 0.0}


def make_optimizer(model, projector_lr, lora_lr):
    """Separate param groups so projector and LoRA get different LRs."""
    projector_params = list(model.projector.parameters())
    projector_ids    = {id(p) for p in projector_params}
    lora_params      = [p for p in model.llm.parameters()
                        if p.requires_grad and id(p) not in projector_ids]

    param_groups = [{"params": projector_params, "lr": projector_lr}]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr})

    return torch.optim.AdamW(param_groups)


def run_epoch(model, dataloader, optimizer, scheduler, grad_clip, stage_label):
    """One training epoch. Returns average loss."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f"Training [{stage_label}]"):
        pixel_values, input_ids, attention_mask, labels = batch

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — stabilises early training
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            grad_clip
        )

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        wandb.log({"train/step_loss": loss.item()})

    return total_loss / len(dataloader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_epochs", type=int,   default=2,
                        help="Epochs to train projector only (Stage 1)")
    parser.add_argument("--stage2_epochs", type=int,   default=3,
                        help="Epochs to train projector + LoRA jointly (Stage 2)")
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--projector_lr",  type=float, default=1e-3,
                        help="LR for the MLP projector")
    parser.add_argument("--lora_lr",       type=float, default=2e-5,
                        help="LR for LoRA params (Stage 2 only)")
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--warmup_steps",  type=int,   default=100)
    parser.add_argument("--llm_model",     type=str,   default="Qwen/Qwen3.5-2B")
    parser.add_argument("--vision_model",  type=str,   default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--mode",          type=str,   default="search",
                        choices=["full", "search"])
    parser.add_argument("--output_dir",    type=str,   default="../results")
    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(project="C5-Week4-CustomLoRA", config=vars(args))

    print("Loading Processors...")
    img_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    tokenizer     = AutoTokenizer.from_pretrained(args.llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Dataset...")
    val_split   = "val_search"   if args.mode == "search" else "val"
    val_ann     = TRAIN_ANN      if args.mode == "search" else VAL_ANN
    val_img_dir = TRAIN_IMG_DIR  if args.mode == "search" else VAL_IMG_DIR

    dataset_train = VizWizDataset(
    TRAIN_ANN, TRAIN_IMG_DIR,
    split='train' if args.mode == 'full' else 'train_search',
    mode=args.mode,          # ← this was missing
    processor=img_processor
    )
    dataset_valid = VizWizDataset(
        val_ann, val_img_dir,
        split=val_split,
        mode=args.mode,          # ← already there but double-check
        processor=img_processor
    )

    wrapper_train = TrainDatasetWrapper(dataset_train, tokenizer)
    wrapper_valid = TrainDatasetWrapper(dataset_valid, tokenizer)

    dataloader_train = DataLoader(wrapper_train, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True)
    dataloader_valid = DataLoader(wrapper_valid, batch_size=1, shuffle=False)

    print("Loading Model...")
    model = CustomVLM(vision_model_name=args.vision_model, llm_name=args.llm_model)

    best_meteor = 0.0
    model_tag   = args.llm_model.replace("/", "_")
    global_epoch = 0

    # ====================================================================== #
    # STAGE 1 — Projector warmup (LoRA frozen)                               #
    # ====================================================================== #
    print("\n========== STAGE 1: Projector-only training ==========")
    model.freeze_llm()

    optimizer_s1 = make_optimizer(model, projector_lr=args.projector_lr, lora_lr=args.lora_lr)
    total_steps_s1 = args.stage1_epochs * len(dataloader_train)
    scheduler_s1 = get_linear_schedule_with_warmup(
        optimizer_s1,
        num_warmup_steps=min(args.warmup_steps, total_steps_s1 // 10),
        num_training_steps=total_steps_s1
    )

    for epoch in range(1, args.stage1_epochs + 1):
        global_epoch += 1
        print(f"\n[Stage 1] Epoch {epoch}/{args.stage1_epochs}")

        avg_loss = run_epoch(model, dataloader_train, optimizer_s1,
                             scheduler_s1, args.grad_clip, "S1-projector")
        print(f"Train Loss: {avg_loss:.4f}")

        metrics = eval_epoch(model, tokenizer, dataloader_valid)
        print(f"Validation: {metrics}")
        wandb.log({
            "train/epoch_loss": avg_loss,
            "val/METEOR":       metrics["METEOR"],
            "epoch":            global_epoch,
            "stage":            1
        })

        if metrics["METEOR"] > best_meteor:
            best_meteor = metrics["METEOR"]
            save_dir = os.path.join(args.output_dir, f"best_custom_vlm_{args.mode}_{model_tag}")
            print(f"New best METEOR: {best_meteor:.2f}! Saving to {save_dir}...")
            os.makedirs(save_dir, exist_ok=True)
            model.llm.save_pretrained(os.path.join(save_dir, "lora_weights"))
            torch.save(model.projector.state_dict(), os.path.join(save_dir, "projector.pt"))
            tokenizer.save_pretrained(save_dir)

    # ====================================================================== #
    # STAGE 2 — Joint projector + LoRA training                              #
    # ====================================================================== #
    print("\n========== STAGE 2: Joint projector + LoRA training ==========")
    model.unfreeze_llm_lora()

    optimizer_s2 = make_optimizer(model, projector_lr=args.projector_lr, lora_lr=args.lora_lr)
    total_steps_s2 = args.stage2_epochs * len(dataloader_train)
    scheduler_s2 = get_linear_schedule_with_warmup(
        optimizer_s2,
        num_warmup_steps=min(args.warmup_steps, total_steps_s2 // 10),
        num_training_steps=total_steps_s2
    )

    for epoch in range(1, args.stage2_epochs + 1):
        global_epoch += 1
        print(f"\n[Stage 2] Epoch {epoch}/{args.stage2_epochs}")

        avg_loss = run_epoch(model, dataloader_train, optimizer_s2,
                             scheduler_s2, args.grad_clip, "S2-joint")
        print(f"Train Loss: {avg_loss:.4f}")

        metrics = eval_epoch(model, tokenizer, dataloader_valid)
        print(f"Validation: {metrics}")
        wandb.log({
            "train/epoch_loss": avg_loss,
            "val/METEOR":       metrics["METEOR"],
            "epoch":            global_epoch,
            "stage":            2
        })

        if metrics["METEOR"] > best_meteor:
            best_meteor = metrics["METEOR"]
            save_dir = os.path.join(args.output_dir, f"best_custom_vlm_{args.mode}_{model_tag}")
            print(f"New best METEOR: {best_meteor:.2f}! Saving to {save_dir}...")
            os.makedirs(save_dir, exist_ok=True)
            model.llm.save_pretrained(os.path.join(save_dir, "lora_weights"))
            torch.save(model.projector.state_dict(), os.path.join(save_dir, "projector.pt"))
            tokenizer.save_pretrained(save_dir)

    print(f"\nTraining complete. Best METEOR: {best_meteor:.2f}")
    wandb.finish()


if __name__ == "__main__":
    main()