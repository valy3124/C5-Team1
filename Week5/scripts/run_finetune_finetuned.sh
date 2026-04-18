#!/bin/bash
#SBATCH --job-name=w5_finetune_finetuned
#SBATCH -p mhigh
#SBATCH -q masterhigh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/finetune_finetuned_%j.out
#SBATCH --error=logs/finetune_finetuned_%j.err

# Fine-tunes the Week4 finetuned BLIP on 3 generated-only dataset variants:
#   4. generated_clusters2S0CFG only
#   5. generated_clusters4S1CFG only
#   6. generated_clusters2S0CFG + 4S1CFG (both)
#
# Validation: always original VizWiz val split.
# Best model saved by METEOR; qualitative samples generated at the end.
#
# To submit: sbatch Week5/scripts/run_finetune_finetuned.sh

source /ghome/group01/miniconda3/bin/activate c5

set -e

FREE_GPU=$(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits \
    | sort -nr | head -n 1 | awk '{print $2}')
export CUDA_VISIBLE_DEVICES=$FREE_GPU
echo "Using GPU $FREE_GPU"

SRC=/ghome/group01/C5/benet/C5-Team1/Week5/src
DATASET=/ghome/group01/C5/dataset
MODELS=/ghome/group01/C5/benet/C5-Team1/Week5/models

# Week4 finetuned checkpoint — processor config is inside best_model/
BASE_MODEL=/ghome/group01/C5/benet/C5-Team1/Week5/models/finetune_blip_both-finetune_full_20260324_220017/best_model

VAL_ANN=$DATASET/VizWiz/annotations/val.json
VAL_IMG=$DATASET/VizWiz/images/val

COMMON_ARGS="--epochs 10 --lr 5e-6 --batch_size 16 --num_workers 4" # original 1e-5

# ── 4/6: generated 2S0CFG only ───────────────────────────────────────────────
echo ""
echo "========================================"
echo " Run 4/6: finetuned BLIP + generated_2S0CFG_only"
echo "========================================"
python $SRC/finetune.py \
    --run_name      finetuned_plus_2S0CFG_only \
    --base_model    "$BASE_MODEL" \
    --train_ann     $DATASET/generated_2S0CFG_only/annotations/train.json \
    --train_img_dir $DATASET/generated_2S0CFG_only/images/train \
    --val_ann       $VAL_ANN \
    --val_img_dir   $VAL_IMG \
    --output_dir    $MODELS \
    $COMMON_ARGS

# ── 5/6: generated 4S1CFG only ───────────────────────────────────────────────
echo ""
echo "========================================"
echo " Run 5/6: finetuned BLIP + generated_4S1CFG_only"
echo "========================================"
python $SRC/finetune.py \
    --run_name      finetuned_plus_4S1CFG_only \
    --base_model    "$BASE_MODEL" \
    --train_ann     $DATASET/generated_4S1CFG_only/annotations/train.json \
    --train_img_dir $DATASET/generated_4S1CFG_only/images/train \
    --val_ann       $VAL_ANN \
    --val_img_dir   $VAL_IMG \
    --output_dir    $MODELS \
    $COMMON_ARGS

# ── 6/6: generated both ──────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Run 6/6: finetuned BLIP + generated_both_only"
echo "========================================"
python $SRC/finetune.py \
    --run_name      finetuned_plus_both_only \
    --base_model    "$BASE_MODEL" \
    --train_ann     $DATASET/generated_both_only/annotations/train.json \
    --train_img_dir $DATASET/generated_both_only/images/train \
    --val_ann       $VAL_ANN \
    --val_img_dir   $VAL_IMG \
    --output_dir    $MODELS \
    $COMMON_ARGS

echo ""
echo "All 3 finetuned-base finetuning runs complete."
