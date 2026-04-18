#!/bin/bash
#SBATCH --job-name=NOT_YOUR_BUSINESS
#SBATCH -p mhigh
#SBATCH -q masterhigh
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sd_explore_%j.out
#SBATCH --error=logs/sd_explore_%j.err

# Source the conda environment 
source /ghome/group01/miniconda3/bin/activate c5

# Make sure we're in the correct directory
cd /ghome/group01/C5/vali/C5-Team1/Week5/src

# Group 6 holds one GPU outside of SLURM, so we asked for 2 and will pick the free one:
FREE_GPU=$(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits | sort -nr | head -n 1 | awk '{print $2}')
export CUDA_VISIBLE_DEVICES=$FREE_GPU

echo "Starting Stable Diffusion Inference Exploration on GPU $FREE_GPU..."
python explore_inference.py
echo "Done!"