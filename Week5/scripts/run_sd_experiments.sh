#!/bin/bash
#SBATCH --job-name=notYourBusiness
#SBATCH -p mlow
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sd_play_%j.out
#SBATCH --error=logs/sd_play_%j.err

source /ghome/group01/miniconda3/bin/activate c5
cd /ghome/group01/C5/vali/C5-Team1/Week5/src

PROMPT="A highly detailed and majestic futuristic city covered in lush green vegetation, glowing neon lights, cinematic lighting, 8k resolution, photorealistic"

MODELS=(
    "runwayml/stable-diffusion-v1-5"
    "stabilityai/stable-diffusion-xl-base-1.0"
    "stabilityai/sdxl-turbo"
)

for MODEL in "${MODELS[@]}"; do
    SAFE_NAME=$(echo $MODEL | cut -d'/' -f2)
    OUTPUT_FILE="../visualizations/${SAFE_NAME}_sample.png"
    
    echo "=================================================="
    echo "Running inference for $MODEL"
    echo "=================================================="
    
    # Set CUDA_VISIBLE_DEVICES explicitly and add PYTORCH_NO_CUDA_MEMORY_CACHING
    # to prevent stale GPU state between model runs
    CUDA_VISIBLE_DEVICES=0 PYTORCH_NO_CUDA_MEMORY_CACHING=1 python generate.py \
        --model_id "$MODEL" \
        --prompt "$PROMPT" \
        --output_path "$OUTPUT_FILE"
    
    echo "Finished $MODEL, exit code: $?"
done

echo "Done!"