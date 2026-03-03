#!/bin/bash
#SBATCH --job-name=detr_baseline
#SBATCH --partition=mlow
#SBATCH --account=master
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

python -m src.inference.run_inference \
  --model detr \
  --exp_name detr_baseline \
  --output src/results/detr_results/detr_baseline_results.jsonl