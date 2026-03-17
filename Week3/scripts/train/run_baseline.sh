#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs/train_baseline_%u_%j.out
#SBATCH -e logs/train_baseline_%u_%j.err

set -e
cd /ghome/group01/C5/benet/C5-Team1/Week3

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

mkdir -p logs results/checkpoints

echo "Starting baseline training on node: $HOSTNAME"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

python train.py \
    --encoder resnet18 \
    --mode search \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-3

echo "Baseline training complete!"
