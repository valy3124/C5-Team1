# Week 5: Image Generation & Datasets

## Overview

This directory includes our work for Week 5, where we focus on exploring image generation models and clustering datasets to generate and analyze custom text-to-image data. Our primary objective is to adapt generating tools to build and curate images using Stable Diffusion models, explore the VizWiz dataset via image embeddings, and evaluate quality.

The main tasks in this module include:

1. **Stable Diffusion Exploration:** Testing and evaluating variations of Stable Diffusion (SD 1.5, SDXL, SDXL Turbo) for different text-to-image capabilities.
2. **Prompt Generation:** Using Qwen LLM models to generate and curate diverse prompts based on our dataset insights.
3. **Dataset Clustering:** Clustering datasets via CLIP embeddings to find subsets and patterns inside VizWiz, along with analyzing data structures.
4. **Generating Datasets:** Using generated prompts to build extended datasets across multiple configurations (e.g., 2-Step and 4-Step CFG tests).
5. **Evaluating Clusters:** Using evaluation scripts like `cluster_metrics.py` to compare properties, structural similarity, and metrics.
6. **Fine-Tuning:** Fine-tuning our best Vision-Language Model from Week 4 (BLIP) using the newly generated synthetic image-caption datasets to observe any downstream performance boosts or impacts.

## Project Structure

```text
Week5/
├── src/
│   ├── clustering.py             # Script to extract CLIP embeddings, perform clustering, and visualise
│   ├── cluster_metrics.py        # Evaluates distance, centroid metrics of our clusters
│   ├── finetune.py               # Fine-tunes BLIP on the generated VizWiz-format datasets
│   ├── generate_images.py        # Generates synthetic images from text prompts
│   ├── generate_prompts_qwen.py  # Generates varied dataset prompts powered by Qwen
│   ├── dataset.py                # Dataset handling
│   ├── explore_inference.py      # Base script testing SD inference models
│   ├── explore_vizwiz.py         # Visual and textual exploration of standard datasets
│   └── process_generated_dataset.py # Pipeline step to organize/post-process generated formats
├── scripts/
│   ├── run_cluster_metrics.sh    # SLURM script to generate and evaluate clusters
│   ├── run_exploration.sh        # SLURM script exploring base generation inferences
│   ├── run_finetune_finetuned.sh # Runs fine-tuning using our best Week 4 checkpoint 
│   ├── run_finetune_pretrained.sh# Runs fine-tuning from a base pretrained BLIP model
│   ├── run_generate_images.sh    # Runs large bulk generation with selected prompts
│   ├── run_generate_prompts.sh   # Automatically generates prompts using Qwen
│   └── run_visualize_generated.sh # Tool to overlay and visually evaluate sets
├── visualizations/               # Output directories containing generated inference comparisons
└── logs/                         # Cluster and SD logging dumps
```

## Exploring & Clustering using Streamlit 

To actively visualize and explore dataset clusters and embeddings (such as CLIP-based clusters in VizWiz), run the Streamlit interface. It allows you to dynamically plot embeddings and interactively inspect clustered groups.

Run the following command (substituting the paths for your specific environment locations):

```bash
streamlit run src/clustering.py -- \
    --step visualize \
    --embeddings_path ../embeddings/clip_embeddings_train.npz \
    --val_embeddings_path ../embeddings/clip_embeddings_val.npz \
    --data_dir /path/to/dataset/train/images \
    --val_data_dir /path/to/dataset/val/images
```

*(Note: Ensure that the `--` is present so `streamlit` passes the remaining arguments to `clustering.py`.)*

## Running Experiments

All commands should be run from the **`Week5/`** directory. 

### Generating Prompts

Generate the synthetic text prompts that will be fed into Stable Diffusion. This relies on the Qwen architecture scripts:

```bash
sbatch scripts/run_generate_prompts.sh
```

### Generating Images

Once prompts are defined, deploy them through the distributed image generator pipeline to build the datasets:

```bash
sbatch scripts/run_generate_images.sh
```

### Processing Data Pipeline

Run post-action scripts to clean captions, filter bad data, or orchestrate dataset configurations:

```bash
sbatch scripts/process_generated_datasets.sh
```

### Fine-Tuning the VLM

Using the combined extended datasets, launch fine-tuning on either our best Week 4 checkpoint or a base pretrained model:

```bash
sbatch scripts/run_finetune_finetuned.sh
sbatch scripts/run_finetune_pretrained.sh
```

### Generating Clustering Metrics

To re-calculate quantitative cluster quality, centroid distances, and general metrics against generated subsets:

```bash
sbatch scripts/run_cluster_metrics.sh
```
