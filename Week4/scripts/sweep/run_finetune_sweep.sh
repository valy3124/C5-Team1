#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs/sweep_finetune_%u_%j.out
#SBATCH -e logs/sweep_finetune_%u_%j.err

SWEEP_ID=$1
COUNT=$2

if [ -z "$SWEEP_ID" ]; then
  echo "Error: Missing Sweep ID."
  echo "Usage: sbatch scripts/sweep/run_finetune_sweep.sh <SWEEP_ID> [COUNT]"
  echo "First run: wandb sweep configs/sweep_finetune.yaml"
  exit 1
fi

set -e
cd /ghome/group01/C5/vali/C5-Team1/Week4

# Activate environment correctly
source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5
mkdir -p logs results

echo "Starting W&B sweep agent on node: $HOSTNAME"
echo "Sweep: $SWEEP_ID"

if [ -z "$COUNT" ]; then
    # Run by default 3 times if no count is provided since we have exactly 3 strategies
    wandb agent --count 3 "c5-team1/C5-Week4-ImageCaptioning/$SWEEP_ID"
else
    wandb agent --count "$COUNT" "c5-team1/C5-Week4-ImageCaptioning/$SWEEP_ID"
fi
