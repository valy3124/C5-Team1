#!/bin/bash
#SBATCH --job-name=generate_clusters
#SBATCH -p mhigh
#SBATCH -q masterhigh
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=13:00:00
#SBATCH --output=logs/generate_clusters_%j.out
#SBATCH --error=logs/generate_clusters_%j.err

source /ghome/group01/miniconda3/bin/activate c5
cd /ghome/group01/C5/vali/C5-Team1/Week5/src

echo "=================================================="
echo "Starting cluster image generation..."
echo "=================================================="

# Set GPU explicitly to ensure a clean state
CUDA_VISIBLE_DEVICES=0 PYTORCH_NO_CUDA_MEMORY_CACHING=1 python generate_images.py

echo "Generation script finished, exit code: $?"
echo "Done!"
