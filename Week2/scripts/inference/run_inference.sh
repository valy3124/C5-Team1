#!/bin/bash
#SBATCH --job-name=yolo_sam_finetuned_inference
#SBATCH --partition=mlow
#SBATCH --account=master
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00


python -m src.inference.run_inference \
    --model grounded_sam \
    --prompt text \
    --text_labels "person. car." \
    --evaluate \
    --dataset kitti_mots \
    --exp_name "grounded_sam_text_person_car_v4" \
    --split validation \
    --save_prompt_boxes