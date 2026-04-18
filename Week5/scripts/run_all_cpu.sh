#!/bin/bash
#SBATCH --job-name=process_datasets_cpu
#SBATCH -p mlow
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/process_datasets_cpu_%j.out
#SBATCH --error=logs/process_datasets_cpu_%j.err

# Runs BOTH visualize and build (all 7 datasets) on CPU only – no GPU requested.
# To submit: sbatch Week5/scripts/run_all_cpu.sh

source /ghome/group01/miniconda3/bin/activate c5

set -e

SRC=/ghome/group01/C5/benet/C5-Team1/Week5/src/process_generated_dataset.py
CSV=/ghome/group01/C5/benet/C5-Team1/Week5/embeddings/cleaned_captions.csv
VIZ=/ghome/group01/C5/benet/C5-Team1/Week5/visualizations
ORIG=/ghome/group01/C5/dataset/VizWiz
OUT=/ghome/group01/C5/dataset

DIR_2S=$VIZ/generated_clusters2S0CFG
DIR_4S=$VIZ/generated_clusters4S1CFG

# ── VISUALIZE ─────────────────────────────────────────────────────────────────
echo "========================================"
echo " VISUALIZE: generated_clusters2S0CFG"
echo "========================================"
python "$SRC" visualize \
    --csv_path "$CSV" \
    --img_dir  "$DIR_2S" \
    --out_dir  "$VIZ/generated_clusters2S0CFG_visualized"

echo ""
echo "========================================"
echo " VISUALIZE: generated_clusters4S1CFG"
echo "========================================"
python "$SRC" visualize \
    --csv_path "$CSV" \
    --img_dir  "$DIR_4S" \
    --out_dir  "$VIZ/generated_clusters4S1CFG_visualized"

# ── BUILD DATASETS ────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " BUILD 2/7: VizWiz + 2S0CFG"
echo "========================================"
python "$SRC" build \
    --csv_path "$CSV" --out_root "$OUT/VizWiz_plus_2S0CFG" \
    --vizwiz_root "$ORIG" --img_dirs "$DIR_2S"

echo ""
echo "========================================"
echo " BUILD 3/7: VizWiz + 4S1CFG"
echo "========================================"
python "$SRC" build \
    --csv_path "$CSV" --out_root "$OUT/VizWiz_plus_4S1CFG" \
    --vizwiz_root "$ORIG" --img_dirs "$DIR_4S"

echo ""
echo "========================================"
echo " BUILD 4/7: VizWiz + 2S0CFG + 4S1CFG"
echo "========================================"
python "$SRC" build \
    --csv_path "$CSV" --out_root "$OUT/VizWiz_plus_both" \
    --vizwiz_root "$ORIG" --img_dirs "$DIR_2S" "$DIR_4S"

echo ""
echo "========================================"
echo " BUILD 5/7: generated_2S0CFG only"
echo "========================================"
python "$SRC" build \
    --csv_path "$CSV" --out_root "$OUT/generated_2S0CFG_only" \
    --img_dirs "$DIR_2S"

echo ""
echo "========================================"
echo " BUILD 6/7: generated_4S1CFG only"
echo "========================================"
python "$SRC" build \
    --csv_path "$CSV" --out_root "$OUT/generated_4S1CFG_only" \
    --img_dirs "$DIR_4S"

echo ""
echo "========================================"
echo " BUILD 7/7: generated both only"
echo "========================================"
python "$SRC" build \
    --csv_path "$CSV" --out_root "$OUT/generated_both_only" \
    --img_dirs "$DIR_2S" "$DIR_4S"

echo ""
echo "All done! Datasets available under $OUT"
