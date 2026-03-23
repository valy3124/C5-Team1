# Week 3: Image Captioning

## Overview

This directory contains the implementations for Week 3, focusing on generating descriptive captions for images using encoder-decoder architectures on the VizWiz dataset. The primary tasks include:

1. **Dataset Handling:** Integrating the VizWiz dataset and testing different tokenization levels (character, subword).
2. **Model Architecture:** Training image captioning models combining CNN vision encoders (ResNet, VGG) with RNN/LSTM decoders.
3. **Hyperparameter Tuning:** Running W&B sweeps to identify the best learning rates, batch sizes, optimizer choices, and other architectural adjustments.
4. **Advanced Extensions:** 
   - Experimenting with CLIP visual backbones as the encoder.
   - Employing xLSTM and other attention-based modifications.
5. **Qualitative & Quantitative Evaluation:** Inspecting the generated text utilizing quantitative metrics (CIDEr, METEOR, BLEU, ROUGE) and qualitatively mapping visual saliency/attention maps to specific caption tokens.

## Dataset

To download and set up the VizWiz dataset, run the provided download script from this directory:

```bash
bash download_vizwiz.sh
```

This will download the annotations and dataset splits into the `dataset/` directory.

## Project Structure

```text
Week3/
├── src/
│   ├── dataset.py                # VizWiz dataset loader and text tokenizers
│   ├── model.py                  # Encoder-Decoder and xLSTM model architectures
│   ├── train.py                  # Main training loop and evaluation pipeline
│   ├── check_dataset.py          # Data exploration and sample checking utilities
│   ├── test_beam.py              # Beam-search evaluation implementation
│   ├── test_saliency.py          # Saliency maps inspection utilities
│   ├── utils/                    # Common functions, plotting, and metric calculation
│   │   ├── compute_baseline_cider.py
│   │   ├── find_example.py
│   │   ├── nltk_metrics.py
│   │   ├── qualitative_metrics.py
│   │   └── saliency_plotter.py
│   └── plots_for_slides/         # Generating results to showcase in presentations
│       ├── plot_qualitative.py
│       ├── plot_text_levels.py
│       ├── plot_all_evolutions.py
│       ├── plot_baseline_vs_final.py
│       └── plot_final_model_simple.py
├── configs/                      # YAML configurations containing hyperparameters
│   ├── baseline.yaml
│   ├── clip_subword_embeddings.yaml
│   └── ...
└── scripts/
    ├── train/                    # SLURM training scripts for isolated runs
    │   ├── run_baseline.sh
    │   ├── run_clip_exp.sh
    │   ├── run_encoder.sh
    │   └── run_train.sh
    └── sweep/                    # SLURM W&B sweep launcher scripts
        ├── create_sweep.sh
        └── run_sweep.sh
```

## Running Experiments

All commands should be run from the **`Week3/`** directory.

### Standard Training

You can train specific models using configuration yaml files or direct command-line arguments using the shell scripts provided. For instance, to train the baseline configuration:

```bash
sbatch scripts/train/run_baseline.sh
```

To run a multi-encoder experiment:
```bash
sbatch scripts/train/run_encoder.sh
```

To experiment with CLIP + subword embeddings:
```bash
sbatch scripts/train/run_clip_exp.sh
```

Alternatively, run the Python code locally using:
```bash
python src/train.py --config configs/baseline.yaml
```

### Hyperparameter Sweeps (Weights & Biases)

We use Weights & Biases (wandb) for sweeping across configurations. Start by defining your sweep:

```bash
bash scripts/sweep/create_sweep.sh
```
*(Copy the generated Sweep ID)*

Then launch a SLURM job to start the W&B agent:
```bash
sbatch scripts/sweep/run_sweep.sh <SWEEP_ID>
```

### Metrics and Evaluation

Training automatically generates history JSONs caching predictions and ground-truth sets. The various quantitative and qualitative tests can be analyzed using our plotting wrappers inside `src/plots_for_slides/`.

For example, to visualize character-vs-subword tokenization performance locally:
```bash
python src/plots_for_slides/plot_text_levels.py
```