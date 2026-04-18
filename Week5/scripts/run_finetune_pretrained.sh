#!/bin/bash
#SBATCH --job-name=w5_finetune_pretrained
#SBATCH -p mlow
#SBATCH -q masterlow
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/finetune_pretrained_%j.out
#SBATCH --error=logs/finetune_pretrained_%j.err

# Fine-tunes BLIP pretrained on 3 VizWiz+generated dataset variants:
#   1. VizWiz + generated_clusters2S0CFG
#   2. VizWiz + generated_clusters4S1CFG
#   3. VizWiz + generated_clusters2S0CFG + generated_clusters4S1CFG
#
# Validation: always original VizWiz val split.
# Best model saved by METEOR; qualitative samples generated at the end.
#
# To submit: sbatch Week5/scripts/run_finetune_pretrained.sh

source /ghome/group01/miniconda3/bin/activate c5

set -e

FREE_GPU=$(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits \
    | sort -nr | head -n 1 | awk '{print $2}')
export CUDA_VISIBLE_DEVICES=$FREE_GPU
echo "Using GPU $FREE_GPU"

SRC=/ghome/group01/C5/benet/C5-Team1/Week5/src
DATASET=/ghome/group01/C5/dataset
MODELS=/ghome/group01/C5/benet/C5-Team1/Week5/models

BASE_MODEL="Salesforce/blip-image-captioning-base"
VAL_ANN=$DATASET/VizWiz/annotations/val.json
VAL_IMG=$DATASET/VizWiz/images/val

COMMON_ARGS="--epochs 10 --lr 2e-5 --batch_size 16 --num_workers 4" # suggested 1e-5 

# ── 1/3: VizWiz + 2S0CFG ─────────────────────────────────────────────────────
# echo ""
# echo "========================================"
# echo " Run 1/3: VizWiz + 2S0CFG"
# echo "========================================"
# python $SRC/finetune.py \
#     --run_name      VizWiz_plus_2S0CFG \
#     --base_model    "$BASE_MODEL" \
#     --train_ann     $DATASET/VizWiz_plus_2S0CFG/annotations/train.json \
#     --train_img_dir $DATASET/VizWiz_plus_2S0CFG/images/train \
#     --val_ann       $VAL_ANN \
#     --val_img_dir   $VAL_IMG \
#     --output_dir    $MODELS \
#     $COMMON_ARGS

# # ── 2/3: VizWiz + 4S1CFG ─────────────────────────────────────────────────────
# echo ""
# echo "========================================"
# echo " Run 2/3: VizWiz + 4S1CFG"
# echo "========================================"
# python $SRC/finetune.py \
#     --run_name      VizWiz_plus_4S1CFG \
#     --base_model    "$BASE_MODEL" \
#     --train_ann     $DATASET/VizWiz_plus_4S1CFG/annotations/train.json \
#     --train_img_dir $DATASET/VizWiz_plus_4S1CFG/images/train \
#     --val_ann       $VAL_ANN \
#     --val_img_dir   $VAL_IMG \
#     --output_dir    $MODELS \
#     $COMMON_ARGS

# ── 3/3: VizWiz + 2S0CFG + 4S1CFG ───────────────────────────────────────────
echo ""
echo "========================================"
echo " Run 3/3: VizWiz + 2S0CFG + 4S1CFG"
echo "========================================"
python $SRC/finetune.py \
    --run_name      VizWiz_plus_both \
    --base_model    "$BASE_MODEL" \
    --train_ann     $DATASET/VizWiz_plus_both/annotations/train.json \
    --train_img_dir $DATASET/VizWiz_plus_both/images/train \
    --val_ann       $VAL_ANN \
    --val_img_dir   $VAL_IMG \
    --output_dir    $MODELS \
    $COMMON_ARGS

echo ""
echo "All 3 pretrained-base finetuning runs complete."
