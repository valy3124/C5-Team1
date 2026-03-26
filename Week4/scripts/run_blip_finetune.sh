#!/bin/bash
#SBATCH --job-name=blip-finetune
#SBATCH --output=logs/finetune_blip_%j.out
#SBATCH --error=logs/finetune_blip_%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -p mlow

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

mkdir -p logs
mkdir -p ../results

echo "Starting BLIP Finetuning..."
# We use strategy 3 (both finetune) to make sure ViT gets trained, full mode
python src/finetune.py --model_type blip --strategy 3 --mode full --epochs 10 --lr 2e-5

echo "Finished BLIP Finetuning!"
