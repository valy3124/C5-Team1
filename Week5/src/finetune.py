"""
finetune.py  –  Week 5 BLIP fine-tuning
========================================
Fine-tunes a BLIP model (pretrained OR a local Week4 checkpoint) on one of
the 6 VizWiz-format dataset variants produced by process_generated_dataset.py.

Validation is always on the original VizWiz val split so all runs are comparable.

Best model is selected by METEOR score; qualitative samples are saved automatically
at the end of training from the best checkpoint.

Example
-------
python finetune.py \\
    --run_name        vizwiz_plus_2S0CFG \\
    --base_model      Salesforce/blip-image-captioning-base \\
    --train_ann       /ghome/group01/C5/dataset/VizWiz_plus_2S0CFG/annotations/train.json \\
    --train_img_dir   /ghome/group01/C5/dataset/VizWiz_plus_2S0CFG/images/train \\
    --val_ann         /ghome/group01/C5/dataset/VizWiz/annotations/val.json \\
    --val_img_dir     /ghome/group01/C5/dataset/VizWiz/images/val \\
    --epochs 10 --lr 2e-5 --batch_size 16 --output_dir ../models
"""

import os
import argparse
import time
import json
import textwrap

import torch
import evaluate
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, AutoProcessor

from dataset import VizWizDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Metrics ──────────────────────────────────────────────────────────────────
bleu   = evaluate.load("bleu")
rouge  = evaluate.load("rouge")
meteor = evaluate.load("meteor")
try:
    cider = evaluate.load("sunhill/cider")
    CIDER_AVAILABLE = True
except Exception as e:
    print(f"CIDEr not available: {e}")
    CIDER_AVAILABLE = False


# ── Dataset wrapper for training (unrolls captions) ──────────────────────────

class TrainDatasetWrapper(Dataset):
    """Wraps VizWizDataset to return (pixel_values, labels) for training.
    Each unique caption becomes its own training sample."""

    def __init__(self, base_dataset: VizWizDataset, tokenizer, max_length=32):
        self.base_dataset = base_dataset
        self.tokenizer    = tokenizer
        self.max_length   = max_length

        self.train_samples = []
        for img_id in base_dataset.valid_image_ids:
            for cap in base_dataset.image_captions.get(img_id, []):
                self.train_samples.append((img_id, cap))

        print(f"[TrainWrapper] {len(self.train_samples)} (image, caption) pairs.")

    def __len__(self):
        return len(self.train_samples)

    def __getitem__(self, idx):
        img_id, caption = self.train_samples[idx]
        img_name  = self.base_dataset.images[img_id]
        img_path  = os.path.join(self.base_dataset.img_dir, img_name)
        img       = PILImage.open(img_path).convert("RGB")
        pixel_values = self.base_dataset.img_proc(img)

        text_inputs = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = text_inputs.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding

        return pixel_values, labels


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Week5 BLIP fine-tuning")
    p.add_argument("--run_name",      required=True,
                   help="Human-readable run label (used in output dir & WandB)")
    p.add_argument("--base_model",    default="Salesforce/blip-image-captioning-base",
                   help="HuggingFace model id OR local path to a BLIP checkpoint")
    # Training split paths
    p.add_argument("--train_ann",     required=True,
                   help="Path to training annotations JSON")
    p.add_argument("--train_img_dir", required=True,
                   help="Path to training images folder")
    # Validation split paths (always original VizWiz val)
    p.add_argument("--val_ann",       required=True,
                   help="Path to validation annotations JSON")
    p.add_argument("--val_img_dir",   required=True,
                   help="Path to validation images folder")
    # Hyper-parameters
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--max_length",  type=int,   default=32)
    # Output
    p.add_argument("--output_dir",  default="../models",
                   help="Root directory for saving results")
    p.add_argument("--qualitative_samples", type=int, default=10,
                   help="Number of qualitative examples to generate at the end")
    return p.parse_args()


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_epoch(model, processor, dataloader):
    model.eval()
    all_preds, all_refs = [], []

    with torch.no_grad():
        for pixel_values, img_ids, _ in tqdm(dataloader, desc="Eval"):
            out_ids = model.generate(
                pixel_values=pixel_values.to(DEVICE), max_new_tokens=30
            )
            preds = [p.strip() for p in
                     processor.batch_decode(out_ids, skip_special_tokens=True)]
            all_preds.extend(preds)
            for img_id in img_ids:
                iid = img_id.item() if hasattr(img_id, "item") else img_id
                all_refs.append(dataloader.dataset.image_captions[iid])

    try:
        b1 = bleu.compute(predictions=all_preds, references=all_refs, max_order=1)
        b2 = bleu.compute(predictions=all_preds, references=all_refs, max_order=2)
        rg = rouge.compute(predictions=all_preds, references=all_refs)
        mt = meteor.compute(predictions=all_preds, references=all_refs)
        cider_score = 0.0
        if CIDER_AVAILABLE:
            cider_score = cider.compute(
                predictions=all_preds, references=all_refs
            ).get("cider_score", 0.0) * 100
        metrics = {
            "BLEU-1":  (b1["bleu"] if b1 else 0.0) * 100,
            "BLEU-2":  (b2["bleu"] if b2 else 0.0) * 100,
            "ROUGE-L": (rg["rougeL"] if rg else 0.0) * 100,
            "METEOR":  (mt["meteor"] if mt else 0.0) * 100,
            "CIDEr":   cider_score,
        }
    except Exception as exc:
        print(f"Metric computation failed: {exc}")
        metrics = {k: 0.0 for k in ["BLEU-1", "BLEU-2", "ROUGE-L", "METEOR", "CIDEr"]}

    model.train()
    return metrics, all_preds, all_refs


# ── Qualitative generation ────────────────────────────────────────────────────

def save_qualitatives(model, processor, val_loader, out_dir: str,
                      n_samples: int = 10):
    """Generate captions for 10 fixed val samples (from Week 4) and save figures."""
    os.makedirs(out_dir, exist_ok=True)
    dataset = val_loader.dataset

    # Fixed filenames from Week 4 to ensure comparability
    fixed_filenames = [
        "VizWiz_val_00000000.jpg",
        "VizWiz_val_00000052.jpg",
        "VizWiz_val_00000102.jpg",
        "VizWiz_val_00000152.jpg",
        "VizWiz_val_00000203.jpg",
        "VizWiz_val_00000254.jpg",
        "VizWiz_val_00000304.jpg",
        "VizWiz_val_00000354.jpg",
        "VizWiz_val_00000404.jpg",
        "VizWiz_val_00000454.jpg"
    ]

    # Map filenames to their internal dataset IDs
    # images is a dict {id: filename}
    reverse_map = {name: iid for iid, name in dataset.images.items()}
    
    model.eval()
    with torch.no_grad():
        for rank, img_name in enumerate(fixed_filenames):
            if img_name not in reverse_map:
                print(f"  [Warning] {img_name} not found in validation dataset – skipping.")
                continue
            
            img_id   = reverse_map[img_name]
            img_path = os.path.join(dataset.img_dir, img_name)

            if not os.path.exists(img_path):
                print(f"  [Warning] {img_path} does not exist – skipping.")
                continue

            pil_img = PILImage.open(img_path).convert("RGB")
            pixel_values = dataset.img_proc(pil_img).unsqueeze(0).to(DEVICE)

            out_ids = model.generate(pixel_values=pixel_values, max_new_tokens=30)
            pred    = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            refs    = dataset.image_captions.get(img_id, ["(no reference)"])

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(pil_img)
            ax.axis("off")
            wrapped_pred = textwrap.fill(f"Pred: {pred}", width=60)
            wrapped_ref  = textwrap.fill(f"Ref:  {refs[0]}", width=60)
            title = wrapped_pred + "\n" + wrapped_ref
            if len(refs) > 1:
                title += "\n" + textwrap.fill(f"Ref2: {refs[1]}", width=60)
            plt.suptitle(title, fontsize=11)
            plt.tight_layout()
            save_path = os.path.join(out_dir, f"sample_{rank:02d}_{img_name}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  Qualitative [{rank+1}/{len(fixed_filenames)}] saved → {save_path}")

    model.train()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── WandB ──
    wandb.init(
        project="C5-Week5-Finetuning",
        name=args.run_name,
        config=vars(args),
    )
    print(f"Config: {vars(args)}")

    # ── Load model ──
    print(f"Loading base model from '{args.base_model}' …")
    processor = AutoProcessor.from_pretrained(args.base_model)
    tokenizer = processor.tokenizer
    model     = BlipForConditionalGeneration.from_pretrained(
        args.base_model, use_safetensors=True
    )
    # Unfreeze everything (strategy 3 equivalent)
    for param in model.parameters():
        param.requires_grad = True
    model.to(DEVICE)

    # ── Datasets ──
    print("Loading training dataset …")
    base_train  = VizWizDataset(
        annotation_file=args.train_ann,
        img_dir=args.train_img_dir,
        split="train",
        mode="full",
        processor=processor,
    )
    train_ds    = TrainDatasetWrapper(base_train, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Loading validation dataset …")
    val_ds = VizWizDataset(
        annotation_file=args.val_ann,
        img_dir=args.val_img_dir,
        split="val",
        mode="full",
        processor=processor,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    # ── Output dir ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(args.output_dir, f"{args.run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    best_model_dir = os.path.join(run_dir, "best_model")
    print(f"Results → {run_dir}")

    best_meteor  = -1.0
    best_epoch   = -1
    best_metrics = {}
    all_epoch_metrics = []

    # ── Training loop ──
    for epoch in range(1, args.epochs + 1):
        print(f"\n── Epoch {epoch}/{args.epochs} ──")
        model.train()
        total_loss = 0.0

        for pixel_values, labels in tqdm(train_loader, desc="Train"):
            pixel_values = pixel_values.to(DEVICE)
            labels       = labels.to(DEVICE)

            input_ids = labels.clone()
            input_ids[input_ids == -100] = tokenizer.pad_token_id
            outputs = model(
                pixel_values=pixel_values, input_ids=input_ids, labels=labels
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            wandb.log({"train/step_loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Train loss: {avg_loss:.4f}")

        # ── Validation ──
        print("Evaluating …")
        metrics, _, _ = eval_epoch(model, processor, val_loader)
        metrics["train_loss"] = avg_loss
        print(f"Val metrics: {metrics}")
        all_epoch_metrics.append({"epoch": epoch, **metrics})

        wandb.log({
            "epoch":             epoch,
            "train/epoch_loss":  avg_loss,
            **{f"val/{k}": v for k, v in metrics.items() if k != "train_loss"},
        })

        # ── Best model ──
        if metrics["METEOR"] > best_meteor:
            best_meteor  = metrics["METEOR"]
            best_epoch   = epoch
            best_metrics = {k: v for k, v in metrics.items()}

            print(f"  ★ New best METEOR {best_meteor:.2f} at epoch {epoch} — saving …")
            model.save_pretrained(best_model_dir)
            processor.save_pretrained(best_model_dir)

            # Save best metrics immediately so they are available even if job is killed
            with open(os.path.join(run_dir, "best_metrics.json"), "w") as f:
                json.dump({"best_epoch": best_epoch, **best_metrics}, f, indent=4)

            # Save qualitative samples from the current best model
            print(f"  Generating {args.qualitative_samples} qualitative samples …")
            save_qualitatives(
                model, processor, val_loader,
                out_dir=os.path.join(run_dir, "visual_samples"),
                n_samples=args.qualitative_samples,
            )

    # ── Save training summary ──
    summary = {
        "run_name":    args.run_name,
        "base_model":  args.base_model,
        "best_epoch":  best_epoch,
        "best_meteor": best_meteor,
        "best_metrics": best_metrics,
        "all_epochs":  all_epoch_metrics,
        "config":      vars(args),
    }
    with open(os.path.join(run_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Also save just the best metrics for easy reading
    with open(os.path.join(run_dir, "best_metrics.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, **best_metrics}, f, indent=4)

    print(f"\n=== Training complete ===")
    print(f"Best epoch: {best_epoch}  |  Best METEOR: {best_meteor:.2f}")
    print(f"Best metrics: {best_metrics}")

    # ── WandB summary ──
    wandb.run.summary["best_epoch"] = best_epoch
    for k, v in best_metrics.items():
        if k != "train_loss":
            wandb.run.summary[f"best_val/{k}"] = v

    wandb.finish()
    print(f"\nAll outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
