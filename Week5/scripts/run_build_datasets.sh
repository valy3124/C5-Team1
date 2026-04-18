#!/bin/bash
#SBATCH --job-name=build_datasets
#SBATCH -p mlow
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_datasets_%j.out
#SBATCH --error=logs/build_datasets_%j.err

# Builds all 7 VizWiz-format dataset variants for fine-tuning.
# Output directories are created under /ghome/group01/C5/dataset/
#
# Datasets built:
#   1. VizWiz                              (original, untouched – just symlinked)
#   2. VizWiz + 2S0CFG
#   3. VizWiz + 4S1CFG
#   4. VizWiz + 2S0CFG + 4S1CFG
#   5. generated_2S0CFG_only
#   6. generated_4S1CFG_only
#   7. generated_both_only (2S0CFG + 4S1CFG)
#
# To submit: sbatch Week5/scripts/run_build_datasets.sh

source /ghome/group01/miniconda3/bin/activate c5

set -e

SRC=/ghome/group01/C5/benet/C5-Team1/Week5/src/process_generated_dataset.py
CSV=/ghome/group01/C5/benet/C5-Team1/Week5/embeddings/cleaned_captions.csv
VIZ=/ghome/group01/C5/benet/C5-Team1/Week5/visualizations
ORIG=/ghome/group01/C5/dataset/VizWiz
OUT=/ghome/group01/C5/dataset

DIR_2S=$VIZ/generated_clusters2S0CFG
DIR_4S=$VIZ/generated_clusters4S1CFG

# ── 1. VizWiz untouched ──────────────────────────────────────────────────────
# The original dataset already lives at $ORIG — no action needed.
echo "========================================"
echo " 1/7  VizWiz (original – already exists at $ORIG)"
echo "========================================"

# ── 2. VizWiz + 2S0CFG ───────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " 2/7  VizWiz + generated_clusters2S0CFG"
echo "========================================"
python "$SRC" build \
    --csv_path    "$CSV" \
    --out_root    "$OUT/VizWiz_plus_2S0CFG" \
    --vizwiz_root "$ORIG" \
    --img_dirs    "$DIR_2S"

# ── 3. generated_2S0CFG_only ─────────────────────────────────────────────────
echo ""
echo "========================================"
echo " 3/7  generated_clusters2S0CFG only"
echo "========================================"
python "$SRC" build \
    --csv_path "$CSV" \
    --out_root "$OUT/generated_2S0CFG_only" \
    --img_dirs "$DIR_2S"

# ── 4. generated_4S1CFG_only ─────────────────────────────────────────────────
# echo ""
# echo "========================================"
# echo " 4/7  generated_clusters4S1CFG only"
# echo "========================================"
# python "$SRC" build \
#     --csv_path "$CSV" \
#     --out_root "$OUT/generated_4S1CFG_only" \
#     --img_dirs "$DIR_4S"

# # ── 5. VizWiz + 4S1CFG ───────────────────────────────────────────────────────
# echo ""
# echo "========================================"
# echo " 5/7  VizWiz + generated_clusters4S1CFG"
# echo "========================================"
# python "$SRC" build \
#     --csv_path    "$CSV" \
#     --out_root    "$OUT/VizWiz_plus_4S1CFG" \
#     --vizwiz_root "$ORIG" \
#     --img_dirs    "$DIR_4S"

# # ── 6. VizWiz + 2S0CFG + 4S1CFG ─────────────────────────────────────────────
# echo ""
# echo "========================================"
# echo " 6/7  VizWiz + 2S0CFG + 4S1CFG (both)"
# echo "========================================"
# python "$SRC" build \
#     --csv_path    "$CSV" \
#     --out_root    "$OUT/VizWiz_plus_both" \
#     --vizwiz_root "$ORIG" \
#     --img_dirs    "$DIR_2S" "$DIR_4S"

# # ── 7. generated_both_only ───────────────────────────────────────────────────
# echo ""
# echo "========================================"
# echo " 7/7  generated_clusters2S0CFG + 4S1CFG only"
# echo "========================================"
# python "$SRC" build \
#     --csv_path "$CSV" \
#     --out_root "$OUT/generated_both_only" \
#     --img_dirs "$DIR_2S" "$DIR_4S"

# echo ""
# echo "All 7 datasets built under $OUT"
