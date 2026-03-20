#!/bin/bash
#SBATCH --job-name=eval_semantic_finetuned
#SBATCH --partition=mlow
#SBATCH --account=master
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/ghome/group01/C5/vali/C5-Team1/Week2/logs/eval_semantic_finetuned_%j.out
#SBATCH --error=/ghome/group01/C5/vali/C5-Team1/Week2/logs/eval_semantic_finetuned_%j.err

set -e
cd /ghome/group01/C5/vali/C5-Team1/Week2

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

SPLIT="validation"

# Give a specific output path to avoid overwriting previous pretrained results
EXP_NAME="semantic_finetuned_sam_${SPLIT}_new"

# Find newest SAM finetune checkpoint from this workspace.
WEIGHTS=$(ls -1t results_finetune/*/*/best_model.pth 2>/dev/null | head -n 1)
if [[ -z "$WEIGHTS" ]]; then
  echo "No best_model.pth found under results_finetune/*/*"
  exit 1
fi

echo "========================================================="
echo " Task : Semantic Segmentation Evaluation (Finetuned ONLY)"
echo " Split: $SPLIT"
echo " Model: $WEIGHTS"
echo " Out  : results_semantic/$EXP_NAME"
echo "========================================================="

echo ""
echo "Running evaluate for finetuned SAM with GT-box prompts..."
python -m src.inference.run_semantic_segmentation \
    --mode finetuned_sam \
    --weights "$WEIGHTS" \
    --split "$SPLIT" \
    --exp_name "$EXP_NAME" \
    --save_viz_every 20

echo ""
echo "Done! Results are in results_semantic/$EXP_NAME/"
