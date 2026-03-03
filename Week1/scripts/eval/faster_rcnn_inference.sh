#!/bin/bash
#SBATCH --job-name=faster_rcnn_baseline
#SBATCH --partition=mlow
#SBATCH --account=master
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

python -m src.inference.run_inference \
  --model faster_rcnn \
  --exp_name faster_rcnn_baseline \
  --output src/results/faster_rcnn_results/faster_rcnn_baseline_results.jsonl