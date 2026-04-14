#!/bin/bash
#SBATCH --job-name=sd_explore
#SBATCH -p mlow
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sd_explore_%j.out
#SBATCH --error=logs/sd_explore_%j.err

# Source the conda environment 
source /ghome/group01/miniconda3/bin/activate c5

# Make sure we're in the correct directory
cd /ghome/group01/C5/vali/C5-Team1/Week5/src

echo "Starting Stable Diffusion Inference Exploration..."
python explore_inference.py
echo "Done!"