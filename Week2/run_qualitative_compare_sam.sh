#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs_eval/qual/qualitative_compare_sam_%j.out
#SBATCH -e logs_eval/qual/qualitative_compare_sam_%j.err

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

FINETUNED_DIR="/ghome/group01/C5/benet/C5-Team1/Week2/results_finetune/final_finetuned/sam_mix_xu4r480y"

python -m src.finetune.qualitative_compare_sam \
    --split validation \
    --finetuned_dir "$FINETUNED_DIR" \
    --n_samples 200 \
    --prompt_type mix \
    --text_prompt "Person. Car" \
    --output_dir results_qualitative_sam/validation_mix
