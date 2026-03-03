#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs/finetune_%u_%j.out
#SBATCH -e logs/finetune_%u_%j.err

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

mkdir -p logs

echo "Starting LoRA run on node: $HOSTNAME"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

python src/fine_tune/fine_tune_detr.py \
    --config src/fine_tune/configs/single_run/config_detr_deart.yaml \
    --lr 0.00005 \
    --epochs 10
