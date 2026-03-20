#!/bin/bash
# Creates a W&B sweep from a config YAML and prints the resulting sweep ID.
# Usage: bash create_sweep.sh <config_yaml>
# Example: bash create_sweep.sh configs/sweep_encoder.yaml

CONFIG=${1:-configs/sweep_encoder.yaml}

cd /ghome/group01/C5/benet/C5-Team1/Week3

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

echo "Creating W&B sweep from: $CONFIG"
wandb sweep "$CONFIG"

echo ""
echo "Copy the sweep ID above and submit the agent with:"
echo "  sbatch scripts/sweep/run_sweep.sh <SWEEP_ID> C5-ImageCaptioning [COUNT]"
