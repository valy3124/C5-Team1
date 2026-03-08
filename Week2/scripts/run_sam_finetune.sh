#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs/finetune_%u_%j.out
#SBATCH -e logs/finetune_%u_%j.err

echo "JOB ID: $SLURM_JOB_ID"

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

# Run BASELINE config
python ../src/finetune/sam_finetune.py --config ../src/finetune/configs/config_sam_base.yaml

echo "Done!"
