#!/bin/bash
# run_sweep_yolo_massive.sh
# Creates the Weights & Biases massive sweep for YOLO multi-stage fine-tuning

# Activate environment if needed
source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Creating MASSIVE W&B sweep for YOLO..."
wandb sweep src/fine_tune/sweep_yolo_massive_stages.yaml

echo ""
echo "Please copy the wandb agent command above (e.g., wandb agent username/project/sweep_id) and run it using the SLURM submission script:"
echo "sbatch run_sweep.sh <SWEEP_ID> YOLO_FineTune_Massive_Stages"
