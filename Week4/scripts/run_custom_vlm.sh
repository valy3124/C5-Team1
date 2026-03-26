#!/bin/bash
#SBATCH --job-name=custom-lora
#SBATCH --output=logs/custom_lora_%j.out
#SBATCH --error=logs/custom_lora_%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -p mlow

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

# Make sure you wait until the BLIP finetune finishes first!
# Substitute "best_model_blip_mode_full_strategy_3" with the actual directory name created!

echo "Training custom VLM built from finetuned BLIP ViT + Qwen2.5 1.5B with LoRA"
python src/train_custom_vlm.py \
    --mode full \
    --vision_model ../results/finetune_blip_both-finetune_full_20260326_015801/best_model \
    --llm_model Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 5 \
    --batch_size 4

echo "Training Complete!"
