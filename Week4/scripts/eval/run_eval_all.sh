#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs/eval_all_%u_%j.out
#SBATCH -e logs/eval_all_%u_%j.err

set -e
cd /ghome/group01/C5/vali/C5-Team1/Week4

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

echo "Starting evaluation on node: $HOSTNAME"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

cd src/

echo "Evaluating vit-gpt2"
python evaluate_pretrained.py --model_type vit-gpt2 --mode full

echo "Evaluating vit-bert"
python evaluate_pretrained.py --model_type vit-bert --mode full

echo "Evaluating blip"
python evaluate_pretrained.py --model_type blip --mode full

echo "Evaluation complete!"
