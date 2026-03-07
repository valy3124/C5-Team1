#!/bin/bash
#SBATCH --job-name=sam_grid_inference
#SBATCH --partition=mlow
#SBATCH --account=master
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00


python -m src.inference.run_inference \
    --model sam \
    --prompt grid \
    --exp_name sam_grid_validation \
    --log_tables