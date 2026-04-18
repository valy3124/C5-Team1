#!/bin/bash
#SBATCH --job-name=visualize_generated
#SBATCH -p mlow
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/visualize_generated_%j.out
#SBATCH --error=logs/visualize_generated_%j.err

# Renders captions below each generated image and saves to mirrored folders.
# To submit: sbatch Week5/scripts/run_visualize_generated.sh

source /ghome/group01/miniconda3/bin/activate c5

set -e

SRC=/ghome/group01/C5/benet/C5-Team1/Week5/src/process_generated_dataset.py
CSV=/ghome/group01/C5/benet/C5-Team1/Week5/embeddings/cleaned_captions.csv
VIZ=/ghome/group01/C5/benet/C5-Team1/Week5/visualizations

# echo "========================================"
# echo " Visualizing: generated_clusters2S0CFG"
# echo "========================================"
# python "$SRC" visualize \
#     --csv_path "$CSV" \
#     --img_dir  "$VIZ/generated_clusters2S0CFG" \
#     --out_dir  "$VIZ/generated_clusters2S0CFG_visualized"

echo ""
echo "========================================"
echo " Visualizing: generated_clusters4S1CFG"
echo "========================================"
python "$SRC" visualize \
    --csv_path "$CSV" \
    --img_dir  "$VIZ/generated_clusters4S1CFG" \
    --out_dir  "$VIZ/generated_clusters4S1CFG_visualized"

echo ""
echo "All visualizations done!"
