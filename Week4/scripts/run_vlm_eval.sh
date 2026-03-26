#!/bin/bash
#SBATCH --job-name=vlm-eval
#SBATCH --output=logs/evaluation_vlm_%j.out
#SBATCH --error=logs/evaluation_vlm_%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -p mhigh   # Switched from mlow to mhigh
#SBATCH -q masterhigh  # Added the QOS flag

# Ensure Conda is initialized
source /ghome/group01/miniconda3/etc/profile.d/conda.sh

# Activate your env
conda activate c5

# Set your Hugging Face token here to access gated models like Llama 3.2
# export HF_TOKEN="your_hf_token_here_starting_with_hf_"

# Make sure logs directory exists
mkdir -p logs
# Run eval for Qwen 3 VL 8B
# echo "Evaluating Qwen2 VL..."
# python src/evaluate_llm.py --model_type qwen2-vl --mode full --output_dir ../results

# # Run eval for Qwen 3 VL 8B
# echo "Evaluating Qwen3 VL 8B..."
# python src/evaluate_llm.py --model_type qwen3-vl-8b --mode full --output_dir ../results

# Run eval for Qwen 3.5 9B
echo "Evaluating Qwen3.5 9B..."
python src/evaluate_llm.py --model_type qwen3.5-9b --mode full --output_dir ../results

echo "Evaluations Complete!"
