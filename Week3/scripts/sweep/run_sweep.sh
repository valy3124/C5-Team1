#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mhigh
#SBATCH --gres gpu:1
#SBATCH -o logs/sweep_%u_%j.out
#SBATCH -e logs/sweep_%u_%j.err

SWEEP_ID=$1
PROJECT=$2
COUNT=$3

if [ -z "$SWEEP_ID" ] || [ -z "$PROJECT" ]; then
  echo "Error: Missing arguments."
  echo "Usage: sbatch run_sweep.sh <SWEEP_ID> <PROJECT_NAME> [COUNT]"
  exit 1
fi

set -e
cd /ghome/group01/C5/xavi/C5-Team1/Week3

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

mkdir -p logs results/checkpoints

echo "Starting W&B sweep agent on node: $HOSTNAME"
echo "Sweep: c5-team1/$PROJECT/$SWEEP_ID"

if [ -z "$COUNT" ]; then
    wandb agent "c5-team1/$PROJECT/$SWEEP_ID"
else
    wandb agent --count "$COUNT" "c5-team1/$PROJECT/$SWEEP_ID"
fi
