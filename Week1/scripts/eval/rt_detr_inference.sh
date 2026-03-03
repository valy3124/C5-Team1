#!/bin/bash
#SBATCH --job-name=rt_detr_baseline
#SBATCH --partition=mlow
#SBATCH --account=master
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

python -m src.inference.run_inference \
  --model rt_detr \
  --exp_name rt_detr_baseline \
  --output src/results/rt_detr_results/rt_detr_baseline_results.jsonl