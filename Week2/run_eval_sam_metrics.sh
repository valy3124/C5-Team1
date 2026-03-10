#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs_eval/quant/eval_sam_metrics_%j.out
#SBATCH -e logs_eval/quant/eval_sam_metrics_%j.err

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

# Run quantitative evaluation on dev split
python -m src.finetune.eval_sam_metrics \
    --split dev \
    --batch_size 4 \
    --prompt_type point \
    --output results_eval/eval_sam_metrics_point_dev.json

# Run quantitative evaluation on validation split
python -m src.finetune.eval_sam_metrics \
    --split validation \
    --batch_size 4 \
    --prompt_type point \
    --output results_eval/eval_sam_metrics_point_validation.json
