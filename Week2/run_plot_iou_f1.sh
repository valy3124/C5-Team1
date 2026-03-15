#!/bin/bash
#SBATCH -n 2
#SBATCH --mem 8G
#SBATCH -p mlow
#SBATCH -o logs_eval/quant/plot_iou_f1_%j.out
#SBATCH -e logs_eval/quant/plot_iou_f1_%j.err

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

python -m src.plots_for_slides.plot_iou_f1 \
    --pretrained results_eval/eval_sam_metrics_mix_validation2.json \
    --finetuned  results_eval/finetuned_comparison_metrics2.json \
    --output     results_eval/plots/
