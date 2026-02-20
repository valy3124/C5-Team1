#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs/sweeps/sweep_%u_%j.out
#SBATCH -e logs/sweeps/sweep_%u_%j.err

SWEEP_ID=$1
PROJECT=$2
COUNT=$3

if [ -z "$SWEEP_ID" ] || [ -z "$PROJECT" ]; then
  echo "Error: Missing arguments."
  echo "Usage: sbatch run_sweep.sh <SWEEP_ID> <PROJECT_NAME> [COUNT]"
  exit 1
fi

# Ensure wandb is available (if using miniconda, activate env)
source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

# Create logs directory if it doesn't exist
mkdir -p logs/sweeps

if [ -z "$COUNT" ]; then
    echo "Starting WandB agent for sweep: $SWEEP_ID in project $PROJECT (Unlimited runs until timeout)"
    wandb agent "c5-team1/$PROJECT/$SWEEP_ID"
else
    echo "Starting WandB agent for sweep: $SWEEP_ID in project $PROJECT (Limit: $COUNT runs)"
    wandb agent --count $COUNT "c5-team1/$PROJECT/$SWEEP_ID"
fi
