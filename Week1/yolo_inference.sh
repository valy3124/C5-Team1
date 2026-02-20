#!/bin/bash
#SBATCH --job-name=yolo_v8_array
#SBATCH --partition=mlow
#SBATCH --account=master
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

WEIGHTS=("yolo26s.pt")
NAMES=("yolo_v26_s_baseline")

python -m src.inference.run_inference \
  --model yolo \
  --weights ${WEIGHTS[$SLURM_ARRAY_TASK_ID]} \
  --exp_name ${NAMES[$SLURM_ARRAY_TASK_ID]} \
  --output src/results/yolo_results/${NAMES[$SLURM_ARRAY_TASK_ID]}_results.jsonl