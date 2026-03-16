#!/bin/bash
#SBATCH --job-name=sam_eval_pretrained
#SBATCH --partition=mlow
#SBATCH --account=master
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/ghome/group01/C5/vali/C5-Team1/Week2/logs/eval_pretrained_%j.out
#SBATCH --error=/ghome/group01/C5/vali/C5-Team1/Week2/logs/eval_pretrained_%j.err

cd /ghome/group01/C5/vali/C5-Team1/Week2

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

# ---- Run 1: Pretrained weights, pretrained protocol (non-zero AP expected) ----
python -m src.inference.eval_pretrained \
    --model_id facebook/sam-vit-base \
    --protocol pretrained \
    --batch_size 4 \
    --output results_eval/pretrained_protocol-pretrained.json

# ---- Run 2: Pretrained weights, finetuning protocol (AP≈0, shows the gap) ----
python -m src.inference.eval_pretrained \
    --model_id facebook/sam-vit-base \
    --protocol finetuned \
    --batch_size 4 \
    --output results_eval/pretrained_protocol-finetuned.json

# ---- Run 3: Finetuned weights, finetuning protocol (matches best_metrics.json) ----
python -m src.inference.eval_pretrained \
    --model_id facebook/sam-vit-base \
    --weights /ghome/group01/C5/vali/C5-Team1/Week2/results_finetune/sam_base_lh35g5yk/best_model.pth \
    --protocol finetuned \
    --batch_size 4 \
    --output results_eval/finetuned_protocol-finetuned.json
