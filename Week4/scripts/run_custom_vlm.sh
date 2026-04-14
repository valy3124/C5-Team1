#!/bin/bash
#SBATCH --job-name=custom-lora
#SBATCH --output=logs/custom_lora_%j.out
#SBATCH --error=logs/custom_lora_%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -p mlow

source /ghome/group01/miniconda3/etc/profile.d/conda.sh
conda activate c5

BEST_MODEL=$1

if [ -z "$BEST_MODEL" ]; then
    echo "Usage: sbatch run_custom_vlm.sh <BEST_MODEL>"
    echo "Example: sbatch run_custom_vlm.sh Qwen2.5-VL-7B-Instruct"
    exit 1
fi

# The path to the previously finetuned BLIP model
VISION_MODEL="/ghome/group01/C5/benet/C5-Team1/Week4/MODELS_ARRANGED/Task1/Task1.2/finetune_blip_both-finetune_full_20260324_220017/best_model"

declare -a MODELS_TO_FINETUNE

# Setup which models to finetune according to the best performing model
case $BEST_MODEL in
    "Qwen2-VL-7B-Instruct")
        MODELS_TO_FINETUNE=("Qwen/Qwen2-VL-2B-Instruct")
        ;;
    "Qwen2.5-VL-7B-Instruct")
        MODELS_TO_FINETUNE=("Qwen/Qwen2.5-VL-3B-Instruct")
        ;;
    "Qwen3-VL-8B-Instruct")
        MODELS_TO_FINETUNE=("Qwen/Qwen3-VL-4B-Instruct" "Qwen/Qwen3-VL-2B-Instruct")
        ;;
    "Qwen3.5-9B")
        MODELS_TO_FINETUNE=("Qwen/Qwen3.5-4B" "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-0.8B")
        ;;
    *)
        echo "Unknown best model configuration string: $BEST_MODEL"
        echo "Valid options: Qwen2-VL-7B-Instruct, Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct, Qwen3.5-9B"
        exit 1
        ;;
esac

OUTPUT_DIR="/ghome/group01/C5/vali/C5-Team1/Week4/MODELS_ARRANGED/results"

for LLM in "${MODELS_TO_FINETUNE[@]}"; do
    echo "======================================================================"
    echo "Finetuning custom VLM with frozen BLIP and $LLM decoder using LoRA..."
    echo "======================================================================"
    python src/train_custom_vlm.py \
        --mode search \
        --vision_model "$VISION_MODEL" \
        --llm_model "$LLM" \
        --output_dir "$OUTPUT_DIR" \
        --stage1_epochs 3 \
        --stage2_epochs 5 \
        --batch_size 4
done

echo "All Training Complete!"
