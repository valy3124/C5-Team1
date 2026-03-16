#!/bin/bash
#SBATCH --job-name=semantic_seg_latest
#SBATCH --partition=mlow
#SBATCH --account=master
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/ghome/group01/C5/vali/C5-Team1/Week2/logs/semantic_seg_latest_%j.out
#SBATCH --error=/ghome/group01/C5/vali/C5-Team1/Week2/logs/semantic_seg_latest_%j.err

set -e
cd /ghome/group01/C5/vali/C5-Team1/Week2

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

SPLIT="validation"

# Find newest SAM finetune checkpoint from this workspace.
WEIGHTS=$(ls -1t results_finetune/sam/*/best_model.pth 2>/dev/null | head -n 1)
if [[ -z "$WEIGHTS" ]]; then
  echo "No best_model.pth found under results_finetune/sam/*"
  exit 1
fi

echo "============================================"
echo " Task (h): Semantic Segmentation Evaluation"
echo " Split   : $SPLIT"
echo " Weights : $WEIGHTS"
echo "============================================"

# ---- 1. Zero-shot: GroundedSAM with text prompts ----
echo ""
echo "[1/5] Text-prompted semantic segmentation (GroundedSAM, zero-shot)"
python -m src.inference.run_semantic_segmentation \
    --mode text_prompt \
    --text_labels "person. car." \
    --split "$SPLIT" \
    --exp_name "semantic_text_prompt_${SPLIT}" \
    --save_viz_every 20

# ---- 2. Pretrained SAM with GT-box prompts ----
echo ""
echo "[2/5] Pretrained SAM with GT-box prompts"
python -m src.inference.run_semantic_segmentation \
    --mode pretrained_sam \
    --split "$SPLIT" \
    --exp_name "semantic_pretrained_sam_${SPLIT}" \
    --save_viz_every 20

# ---- 3. Finetuned SAM with GT-box prompts ----
echo ""
echo "[3/5] Finetuned SAM with GT-box prompts"
python -m src.inference.run_semantic_segmentation \
    --mode finetuned_sam \
    --weights "$WEIGHTS" \
    --split "$SPLIT" \
    --exp_name "semantic_finetuned_sam_${SPLIT}" \
    --save_viz_every 20

# ---- 4. Rich open-vocabulary text prompting (qualitative only) ----
echo ""
echo "[4/5] Rich open-vocabulary semantic segmentation (qualitative demo)"
python -m src.inference.run_semantic_segmentation \
    --mode rich_text_prompt \
    --split "$SPLIT" \
    --exp_name "semantic_rich_text_prompt_${SPLIT}" \
    --save_viz_every 10

# ---- 5. Text prompt with synonym-enriched labels ----
echo ""
echo "[5/5] Text-prompted with synonym labels (GroundedSAM, zero-shot)"
python -m src.inference.run_semantic_segmentation \
    --mode text_prompt \
    --text_labels "person . pedestrian . man . woman . cyclist . human . car . vehicle . van . truck . bus ." \
    --split "$SPLIT" \
    --exp_name "semantic_text_prompt_synonyms_${SPLIT}" \
    --save_viz_every 20

echo ""
echo "All runs complete. Results in results_semantic/"
