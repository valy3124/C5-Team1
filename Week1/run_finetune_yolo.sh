#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs/finetune_%u_%j.out
#SBATCH -e logs/finetune_%u_%j.err

# Activate environment
source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

# Run training
python -m src.fine_tune.fine_tune_yolo --config src/fine_tune/config_example_yolo.yaml
