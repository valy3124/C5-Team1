#!/bin/bash
#SBATCH --job-name=blip-finetune
#SBATCH --output=logs/finetune_blip_%j.out
#SBATCH --error=logs/finetune_blip_%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=24G
#SBATCH -p mhigh
#SBATCH -q masterhigh

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

mkdir -p logs
mkdir -p ../results

# Group 6 holds one GPU outside of SLURM, so we asked for 2 and will pick the free one:
FREE_GPU=$(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits | sort -nr | head -n 1 | awk '{print $2}')
export CUDA_VISIBLE_DEVICES=$FREE_GPU

echo "Starting BLIP Finetuning on GPU $FREE_GPU..."
# We use strategy 3 (both finetune) to make sure ViT gets trained, full mode
python src/finetune.py --model_type blip --strategy 3 --mode full --epochs 10 --lr 2e-5

echo "Finished BLIP Finetuning!"
