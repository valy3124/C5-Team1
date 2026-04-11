# Week 4: Vision-Language Models and PEFT

## Overview

This directory includes our work for Week 4, where we focus on building, training, and exploring custom Vision-Language Models (VLMs) and applying Parameter-Efficient Fine-Tuning (PEFT) techniques. Our primary objective is to adapt powerful pre-trained LLMs to understand visual contexts and generate domain-specific image captions for the VizWiz dataset.

The main tasks in this module include:

1. **Pre-trained VLM baselines & Fine-tuning:** Evaluating strong out-of-the-box Vision-Language Models (like **BLIP**, **ViT-GPT2** and **ViT-Bert**) zero-shot on VizWiz, and running full fine-tuning routines to set a baseline performance.
2. **Custom VLM Architecture:** Developing a custom model combining a frozen **BLIP** vision encoder with a powerful language decoder (e.g., **Qwen/Qwen3.5** in sizes like 0.8B, 2B, and 4B).
3. **MLP Projector Mapping:** Utilizing a trainable Multi-Layer Perceptron (MLP) or Linear layer to project the visual embeddings from the frozen encoder into the LLM's embedding space.
4. **Two-Stage Training with PEFT:** 
   - **Stage 1:** Training only the MLP projector to bridge the modality gap.
   - **Stage 2:** Applying Low-Rank Adaptation (LoRA) to key LLM mechanisms (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and jointly training the projector and the LoRA adapters while keeping the rest of the LLM frozen to avoid catastrophic forgetting and Out-of-Memory (OOM) issues.
5. **Qualitative & Quantitative Evaluation:** Developing evaluation pipelines to accurately track CIDEr and METEOR metrics, handling multi-device tensors correctly.
6. **Visualizations & Edge-cases:** A detailed qualitative visualization script comparing VLM outputs iteratively against Ground Truth strings, including inference testing on edge-cases like blank/black images.

## Project Structure

```text
Week4/
├── src/
│   ├── dataset.py                # VizWiz dataset handler customized for PEFT VLMs
│   ├── train_custom_vlm.py       # Main script for two-stage custom VLM training
│   ├── visualize_custom_vlm.py   # Script to generate qualitative text-overlay visualisations
│   ├── finetune.py               # Pre-built model fine-tuning utilities
│   ├── evaluate_llm.py           # Baseline evaluator for standard LLMs/VLMs
│   ├── evaluate_finetuned.py     # Evaluation loops for trained VLM variants
│   ├── evaluate_pretrained.py    # Zero-shot inference metrics on base models
│   └── analyze_vizwiz_stats.py   # Dataset insights
├── configs/                      # YAML configurations containing hyperparameter sweeps
│   └── sweep_finetune.yaml
└── scripts/
    ├── run_custom_vlm.sh         # SLURM script to train custom VLM variants
    ├── run_visualize.sh          # SLURM script iterating thru custom models (0.8B, 2B, 4B) for eval
    ├── run_vlm_eval.sh           # Quantitative evaluation SLURM launcher
    ├── run_blip_finetune.sh      # Baseline finetuning launcher
    ├── eval/                     # Targeted sub-evaluation batch scripts
    └── sweep/                    # SLURM W&B sweep scripts
```

## Running Experiments

All commands should be run from the **`Week4/`** directory to ensure paths align. 

### Evaluating Pre-trained Baselines

To calculate zero-shot inference metrics or evaluate baseline text-to-image models like BLIP against the VizWiz validation set:

```bash
# Evaluate out-of-the-box pre-trained performance
sbatch scripts/eval/run_pretrained_eval.sh # (Or run python src/evaluate_pretrained.py directly)
```

To run fine-tuning jobs to adjust these base models more closely to the VizWiz domain:

```bash
sbatch scripts/run_blip_finetune.sh
```

### Training the Custom VLM

You can launch the training for our two-stage Qwen architecture using the provided batch script. It maps PyTorch tensor operations directly and handles switching from frozen Stage 1 models into LoRA Stage 2 models on the HPC block:

```bash
sbatch scripts/run_custom_vlm.sh
```

### Qualitative Visualizations

To render qualitative visualization tiles (overlaying the image with our custom VLM predicted captions vs the Ground Truth captions), you can launch the visualization suite. The batch script iterates properly over the 0.8B, 2B, and 4B variations to compare outputs:

```bash
sbatch scripts/run_visualize.sh
```

The script accurately maps against the `full` evaluation dataset mode and tests how the models perform against standard inputs as well as fully black-image anomalies.

### Quantitative Evaluation

To calculate standard validation metrics on complete test-sets for any tuned or custom VLM, execute:

```bash
sbatch scripts/run_vlm_eval.sh
```

## Artifacts & Logs

* Output artifacts and metrics are logged via stdout/stderr into the `logs/` directory and synchronized with our **Weights & Biases (W&B)** setups when configured.
* High-performing checkpoints and custom adapter weights (`lora_weights`) are tracked within `MODELS_ARRANGED/results/`.
* High-resolution tile outputs from `visualize_custom_vlm.py` are saved in the `visualizations/{MODEL_NAME}/` folder.