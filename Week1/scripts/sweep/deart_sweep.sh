#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs/finetune_%u_%j.out
#SBATCH -e logs/finetune_%u_%j.err

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

# 2. Make sure the logs directory exists so SLURM doesn't crash
mkdir -p logs

# 3. Log the node info to help with debugging
echo "Starting W&B Sweep Agent on node: $HOSTNAME"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"


wandb agent --count 1 c5-team1/C5-Domain-Shift/9up2om2e