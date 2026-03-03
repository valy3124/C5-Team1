#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs/nms_eval_%u_%j.out
#SBATCH -e logs/nms_eval_%u_%j.err

# Activate conda environment
source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

EXP_DIR="/ghome/group01/C5/benet/C5-Team1/Week1/results/faster_rcnn/faster_rcnn_Aug_color_jitter_jc8d6k1d"

python src/fine_tune/eval_nms_impact.py --exp_dir "$EXP_DIR"
