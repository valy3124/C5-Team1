#!/bin/bash
#SBATCH --job-name=yolo_v8_array
#SBATCH --partition=mlow
#SBATCH --account=master
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

WEIGHTS=("yolov10b.pt")
NAMES=("yolo_v10_b")

python -m src.inference.run_inference \
  --model yolo \
  --weights ${WEIGHTS[$SLURM_ARRAY_TASK_ID]} \
  --exp_name ${NAMES[$SLURM_ARRAY_TASK_ID]} \
  --output src/results/yolo_results/${NAMES[$SLURM_ARRAY_TASK_ID]}_results.jsonl