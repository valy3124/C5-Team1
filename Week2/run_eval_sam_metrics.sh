#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs_eval/quant/eval_sam_metrics_%j.out
#SBATCH -e logs_eval/quant/eval_sam_metrics_%j.err

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

# Run quantitative evaluation with mixed prompts (bbox + point + text) on validation split
python -m src.finetune.eval_sam_metrics \
    --split validation \
    --batch_size 4 \
    --prompt_type mix \
    --text_prompt "Person. Car" \
    --output results_eval/eval_sam_metrics_mix_validation.json
