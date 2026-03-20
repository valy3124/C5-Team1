#!/bin/bash
#SBATCH --job-name=sam_qualitative
#SBATCH --partition=mlow
#SBATCH --account=master
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/ghome/group01/C5/vali/C5-Team1/Week2/logs/qualitative_%j.out
#SBATCH --error=/ghome/group01/C5/vali/C5-Team1/Week2/logs/qualitative_%j.err

# Run from Week2/ directory
cd /ghome/group01/C5/vali/C5-Team1/Week2

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

python -m src.inference.qualitative_compare \
    --finetuned_weights results_finetune/sam_base_lh35g5yk/best_model.pth \
    --n_samples 30 \
    --prompt center_bb_gt \
    --split validation \
    --skip_empty \
    --output_dir results_qualitative/pretrained_vs_finetuned
