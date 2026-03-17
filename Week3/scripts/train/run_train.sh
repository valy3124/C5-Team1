#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs/train_%u_%j.out
#SBATCH -e logs/train_%u_%j.err

set -e
cd /ghome/group01/C5/vali/C5-Team1/Week3

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

mkdir -p logs results/checkpoints

echo "Starting training on node: $HOSTNAME"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Default single-run: change --encoder to any of: resnet18 resnet34 resnet50 vgg16 vgg19
python train.py \
    --encoder resnet18 \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-3

echo "Done!"
