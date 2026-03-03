#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs/eval_%u_%j.out
#SBATCH -e logs/eval_%u_%j.err

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

mkdir -p logs

echo "Starting Zero-Shot Evaluation..."

# Notice the new --eval_only flag!
python src/fine_tune/fine_tune_detr.py \
    --config src/fine_tune/config_detr_deart.yaml \
    --eval_only