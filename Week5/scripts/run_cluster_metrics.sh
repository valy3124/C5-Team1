#!/bin/bash
#SBATCH --job-name=cluster_metrics
#SBATCH -p mlow
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cluster_metrics_%j.out
#SBATCH --error=logs/cluster_metrics_%j.err

# Source the conda environment 
source /ghome/group01/miniconda3/bin/activate c5

# Optional: Add any pre-requisites
SPLIT=$1
if [ -z "$SPLIT" ]; then
    echo "Usage: sbatch run_cluster_metrics.sh <train|val>"
    exit 1
fi

if [ "$SPLIT" == "train" ]; then
    CSV_PATH="../embeddings/train_clusters_clip_hdbscan_mcs15_ms5_eom_umap_cosine_2d.csv"
else
    CSV_PATH="../embeddings/val_clusters_clip_hdbscan_mcs15_ms5_eom_umap_cosine_2d.csv"
fi

echo "Running cluster metrics evaluation for split: $SPLIT"
echo "Using CSV: $CSV_PATH"

python ../src/cluster_metrics.py --split $SPLIT --csv_path "$CSV_PATH"
