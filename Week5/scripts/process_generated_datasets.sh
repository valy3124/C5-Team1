#!/bin/bash
#SBATCH --job-name=process_generated
#SBATCH -p mlow
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/process_generated_%j.out
#SBATCH --error=logs/process_generated_%j.err

# process_generated_datasets.sh
# Runs process_generated_dataset.py in both modes for both generated
# image directories (generated_clusters2S0CFG and generated_clusters4S1CFG).
#
# To submit: sbatch Week5/scripts/process_generated_datasets.sh

source /ghome/group01/miniconda3/bin/activate c5

set -e  # exit on first error

SCRIPT_DIR="/ghome/group01/C5/benet/C5-Team1/Week5/scripts"
SRC_DIR="$SCRIPT_DIR/../src"
EMBEDDINGS="$SCRIPT_DIR/../embeddings/cleaned_captions.csv"
VIZ_ROOT="$SCRIPT_DIR/../visualizations"
VIZWIZ_ROOT="$SCRIPT_DIR/../VizWiz"

IMG_DIRS=(
    "generated_clusters2S0CFG"
    "generated_clusters4S1CFG"
)

# ─────────────────────────────────────────────
# Mode 1 – visualize (caption overlaid on image)
# ─────────────────────────────────────────────
echo "========================================"
echo " MODE: visualize"
echo "========================================"

for DIR_NAME in "${IMG_DIRS[@]}"; do
    IMG_DIR="$VIZ_ROOT/$DIR_NAME"
    OUT_DIR="$VIZ_ROOT/${DIR_NAME}_visualized"

    echo ""
    echo "  Input : $IMG_DIR"
    echo "  Output: $OUT_DIR"

    python "$SRC_DIR/process_generated_dataset.py" visualize \
        --csv_path   "$EMBEDDINGS" \
        --img_dir    "$IMG_DIR"    \
        --out_dir    "$OUT_DIR"
done

# ─────────────────────────────────────────────
# Mode 2 – merge into VizWiz train split
# ─────────────────────────────────────────────
# echo ""
# echo "========================================"
# echo " MODE: merge → VizWiz/images/train/"
# echo "========================================"

# for DIR_NAME in "${IMG_DIRS[@]}"; do
#     IMG_DIR="$VIZ_ROOT/$DIR_NAME"

#     echo ""
#     echo "  Input : $IMG_DIR"
#     echo "  Target: $VIZWIZ_ROOT"

#     python "$SRC_DIR/process_generated_dataset.py" merge \
#         --csv_path    "$EMBEDDINGS" \
#         --img_dir     "$IMG_DIR"    \
#         --vizwiz_root "$VIZWIZ_ROOT"
# done

# echo ""
# echo "All done!"
