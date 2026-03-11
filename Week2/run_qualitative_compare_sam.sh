#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs_eval/qual/qualitative_compare_sam_%j.out
#SBATCH -e logs_eval/qual/qualitative_compare_sam_%j.err

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

# FINETUNED_DIR="/ghome/group01/C5/benet/C5-Team1/Week2/results_finetune/final_finetuned/sam_bbox_j6cstc09"
    # --prompt_type bbox \

# FINETUNED_DIR="/ghome/group01/C5/benet/C5-Team1/Week2/results_finetune/final_finetuned/sam_point_2oc8fzte"
#     # --prompt_type point \

FINETUNED_DIR="/ghome/group01/C5/benet/C5-Team1/Week2/results_finetune/sam_vit_b/sam_no_aug_text_LR5.0e-05_Opt_AdamW_GC_1.0_cojcfscd"
    # --prompt_type text \
    # --text_prompt "Person. Car" \

# Generate qualitative comparison for validation split
python -m src.finetune.qualitative_compare_sam \
    --split validation \
    --finetuned_dir "$FINETUNED_DIR" \
    --n_samples 200 \
    --prompt_type text \
    --text_prompt "Person. Car" \
    --output_dir results_qualitative_sam/validation_text
