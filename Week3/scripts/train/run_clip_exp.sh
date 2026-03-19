#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs-text-level/clip_subword_%j.out
#SBATCH -e logs-text-level/clip_subword_%j.err

set -e
cd /ghome/group01/C5/benet/C5-Team1/Week3

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

echo "Starting baseline training on node: $HOSTNAME"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

python train.py --config configs/clip_subword_embeddings.yaml

echo "Baseline training complete!"