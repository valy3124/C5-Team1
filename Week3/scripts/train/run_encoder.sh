#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs/encoder_%u_%j.out
#SBATCH -e logs/encoder_%u_%j.err

# Usage:
#   sbatch scripts/train/run_encoder.sh          (uses encoder from encoder_single.yaml)
#   sbatch scripts/train/run_encoder.sh vgg19    (overrides encoder at runtime)

ENCODER=$1

set -e
cd /ghome/group01/C5/benet/C5-Team1/Week3

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

mkdir -p logs results/encoder_sweep

echo "Starting encoder training on node: $HOSTNAME"
if [ -z "$ENCODER" ]; then
    echo "Using configuration from: configs/encoder_single.yaml"
else
    echo "Encoder override: $ENCODER"
fi

python train.py \
    --config configs/encoder_single.yaml \
    ${ENCODER:+--encoder "$ENCODER"}

echo "Training complete!"
