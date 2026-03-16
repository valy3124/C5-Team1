# Week 2: Object Segmentation

## Overview

This directory contains the implementations for Week 2, which focuses on:
1. **Prompt-based Object Discovery** using GroundingDINO to automatically detect bounding boxes from free-form text labels (e.g., "car. pedestrian.").
2. **Zero-Shot Segmentation** using the Segment Anything Model (SAM) prompted by GroundingDINO bounding boxes to achieve precise instance and semantic segmentation without prior training.
3. **Fine-Tuning SAM's Mask Decoder** on the KITTI-MOTS dataset to improve boundary predictions for our specific classes while retaining SAM's zero-shot visual encoder features.
4. **Generalization Analysis** by comparing zero-shot and fine-tuned results on the out-of-domain DEArt dataset.

## Project Structure

```
Week2/
├── src/
│   ├── datasets.py                   # KITTI-MOTS wrapper and mask/bbox augmented pipelines
│   ├── models/                       # Model wrappers (Grounded SAM, YOLO-SAM)
│   ├── inference/                    # Evaluation, qualitative comparison, DEArt inference scripts
│   ├── finetune/                     # SAM fine-tuning suite and configs
│   │   ├── configs/                  # YAML configurations (base, mix, point, text, bbox)
│   │   ├── sam_finetune.py           # Training loop for SAM Mask Decoder
│   │   └── utils.py                  # Shared training utilities
│   ├── prompting/                    # Different prompting strategies (Center BB GT, Grid, SIFT)
│   └── visualizations/               # Scripts to generate plots, gifs, and slide metrics
└── scripts/
    ├── train/                        # SLURM training jobs for SAM model
    ├── eval/                         # SLURM evaluation, testing, and metric generation scripts
    ├── inference/                    # SLURM qualitative and output visualization batch jobs
    └── utils/                        # Downloading scripts (e.g., DEArt fetching)
```

## Running Experiments

All commands should normally be run from the **`Week2/`** directory.

### Training / Fine-tuning SAM

We fine-tune the SAM Mask Decoder using different types of prompts (Points, BBoxes, Text, or a uniform Mix during training). Each strategy has an associated configuration file:

```bash
sbatch scripts/train/run_sam_finetune.sh
```
*(You can easily modify the config loaded within the bash script between `config_sam_mix.yaml`, `config_sam_bboxes.yaml`, `config_sam_text.yaml`, etc.)*

### Evaluating Models

To evaluate the pretrained Baseline SAM using box prompts:
```bash
sbatch scripts/eval/run_eval_pretrained.sh
```

To run detailed COCO instance segmentation evaluations across prompt types for finetuned weights:
```bash
sbatch scripts/eval/run_eval_sam_metrics.sh
```

### Visualizations and Qualitative Comparisons

It is helpful to visually inspect the difference the fine-tuning achieved. To generate the side-by-side (Pretrained vs Finetuned) comparison strips used in our slides:
```bash
sbatch scripts/inference/run_qualitative_compare.sh
```

To run Grounded SAM text-prompted segmentation on the generic domain DEArt dataset to test zero-shot capabilities and fine-tuning catastrophic forgetting:
```bash
sbatch scripts/inference/run_deart_inference.sh
```
