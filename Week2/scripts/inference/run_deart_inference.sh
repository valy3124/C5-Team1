#!/bin/bash
#SBATCH --job-name=deart_domain_shift
#SBATCH --partition=mlow
#SBATCH --account=master
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/ghome/group01/C5/vali/C5-Team1/Week2/logs/deart_%j.out
#SBATCH --error=/ghome/group01/C5/vali/C5-Team1/Week2/logs/deart_%j.err

set -e
cd /ghome/group01/C5/vali/C5-Team1/Week2

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

# Finetuned SAM weights (GT-box protocol, trained on KITTI-MOTS)
WEIGHTS="results_finetune/sam_bbox_j6cstc09/best_model.pth"

SPLIT="validation"
DEART_ROOT="DEArt"

echo "============================================"
echo " Task (f): SAM Domain-Shift on DEArt"
echo " Split  : $SPLIT"
echo " Dataset: $DEART_ROOT"
echo "============================================"

# ---- Approach 2a: Pretrained SAM + GT boxes --------------------------------
echo ""
echo "[1/3] Pretrained SAM with GT bounding-box prompts"
python -m src.inference.run_deart_inference \
    --mode gt_box_pretrained \
    --root "$DEART_ROOT" \
    --split "$SPLIT" \
    --exp_name "deart_gt_box_pretrained_${SPLIT}" \
    --save_viz_every 50

# ---- Approach 2b: KITTI-MOTS Finetuned SAM + GT boxes ----------------------
echo ""
echo "[2/3] KITTI-MOTS Finetuned SAM with GT bounding-box prompts"
python -m src.inference.run_deart_inference \
    --mode gt_box_finetuned \
    --weights "$WEIGHTS" \
    --root "$DEART_ROOT" \
    --split "$SPLIT" \
    --exp_name "deart_gt_box_finetuned_${SPLIT}" \
    --save_viz_every 50

# ---- Approach 1a: GroundedSAM (pretrained) + DeART text labels -------------
echo ""
echo "[3/3] Pretrained GroundedSAM with DeART class-name text prompts"
python -m src.inference.run_deart_inference \
    --mode text_prompt_pretrained \
    --root "$DEART_ROOT" \
    --split "$SPLIT" \
    --exp_name "deart_text_prompt_pretrained_${SPLIT}" \
    --save_viz_every 50

# ---- Approach 1b: Finetuned SAM (text-prompt trained) + DeART text labels -
echo ""
echo "[4/4] KITTI-MOTS text-finetuned SAM with DeART class-name text prompts"
python -m src.inference.run_deart_inference \
    --mode text_prompt_finetuned \
    --weights "results_finetune/sam_text_cojcfscd/best_model.pth" \
    --root "$DEART_ROOT" \
    --split "$SPLIT" \
    --exp_name "deart_text_prompt_finetuned_${SPLIT}" \
    --save_viz_every 50

echo ""
echo "All runs complete.  Results in results_deart/"
