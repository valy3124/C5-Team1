#!/bin/bash
#SBATCH --job-name=qwen_prompts
#SBATCH -p mlow
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/qwen_prompts_%j.out
#SBATCH --error=logs/qwen_prompts_%j.err

# Source the conda environment 
source /ghome/group01/miniconda3/bin/activate c5

echo "Starting Qwen Prompt Generation..."

# Run the prompt generation script
python ../src/generate_prompts_qwen.py \
    --csv_path ../embeddings/cluster_prompts.csv \
    --output_path ../embeddings/qwen_outcomes.csv \
    --model_id "Qwen/Qwen3.5-2B"

echo "Finished generating prompts."

echo "Cleaning generated captions..."
python ../src/clean_captions.py \
    --csv_path ../embeddings/qwen_outcomes.csv \
    --output_path ../embeddings/cleaned_captions.csv

echo "Finished cleaning captions."
