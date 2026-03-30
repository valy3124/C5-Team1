#!/bin/bash
#SBATCH --job-name=vlm-eval
#SBATCH --output=logs/evaluation_vlm_%j.out
#SBATCH --error=logs/evaluation_vlm_%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -p mlow   # Switched from mlow to mhigh
#SBATCH -q masterhigh  # Added the QOS flag

# Ensure Conda is initialized
source /ghome/group01/miniconda3/etc/profile.d/conda.sh

# Activate your env
conda activate c5

# Set your Hugging Face token here to access gated models like Llama 3.2
# export HF_TOKEN="your_hf_token_here_starting_with_hf_"

# Make sure logs directory exists
mkdir -p logs
# Run eval for all requested Qwen Models
OUTPUT_DIR="/ghome/group01/C5/benet/C5-Team1/Week4/MODELS_ARRANGED/results"

echo "Evaluating Qwen2-VL-7B-Instruct..."
python src/evaluate_llm.py --model_type Qwen2-VL-7B-Instruct --mode full --output_dir $OUTPUT_DIR

echo "Evaluating Qwen2.5-VL-7B-Instruct..."
python src/evaluate_llm.py --model_type Qwen2.5-VL-7B-Instruct --mode full --output_dir $OUTPUT_DIR

echo "Evaluating Qwen3-VL-8B-Instruct..."
python src/evaluate_llm.py --model_type Qwen3-VL-8B-Instruct --mode full --output_dir $OUTPUT_DIR

echo "Evaluating Qwen3.5-9B..."
python src/evaluate_llm.py --model_type Qwen3.5-9B --mode full --output_dir $OUTPUT_DIR

echo "Evaluations Complete!"
