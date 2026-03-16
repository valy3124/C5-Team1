#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs_eval/quant/batch_eval_finetuned_%j.out
#SBATCH -e logs_eval/quant/batch_eval_finetuned_%j.err

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

# Run batch evaluation on all finetuned models
# This will output a JSON comparing all models in results_finetune/final_finetuned
python -m src.inference.batch_eval_finetuned \
    --split validation \
    --batch_size 4 \
    --output results_eval/finetuned_comparison_metrics2.json
