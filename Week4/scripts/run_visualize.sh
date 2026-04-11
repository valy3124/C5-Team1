#!/bin/bash
#SBATCH --job-name=viz-custom
#SBATCH --output=logs/viz_custom_%j.out
#SBATCH --error=logs/viz_custom_%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -p mhigh
#SBATCH -q masterhigh

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

MODE=${1:-"full"}

for SIZE in "0.8B" "2B" "4B"; do
    MODEL_PATH="MODELS_ARRANGED/results/best_custom_vlm_search_Qwen_Qwen3.5-${SIZE}"
    LLM_MODEL="Qwen/Qwen3.5-${SIZE}"

    echo "======================================================================"
    echo "Generating visualizations on GPU for: $MODEL_PATH"
    echo "LLM: $LLM_MODEL | Mode: $MODE"
    echo "======================================================================"

    python src/visualize_custom_vlm.py \
        --model_path "$MODEL_PATH" \
        --llm_model "$LLM_MODEL" \
        --mode "$MODE"
done

echo "Done!"
