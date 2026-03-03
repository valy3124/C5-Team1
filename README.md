# MCV C5 Project (Group 1)

## Members

- Diego Hernández Antón
- Oriol Juan Sabater
- Valentin Micu Hontan
- Xavier Pacheco Bach
- Benet Ramió Comas

## Quick Start

`setup.sh` automates the installation of Miniconda, creates the environment, and installs all necessary library dependencies:

```bash
./setup.sh
```

Then, to work within the environment:

```bash
conda activate c5
```

If you need to update dependencies, add new packages to `requirements.txt` and run:

```bash
pip install -r requirements.txt
```

### Dataset

To download the DEArt dataset for task f, run the provided script from the `Week1/src/` directory:

```bash
cd Week1/src
bash download_deart.sh
```

This will download and place the dataset under `Week1/src/DEArt/`.

---

## Project Structure

```
Week1/
├── src/
│   ├── datasets.py                   # KITTI-MOTS and DEArt dataset wrappers
│   ├── deart_stats.py                # DEArt dataset statistics and plots
│   ├── models/                       # Model definitions (DETR, Faster R-CNN, RT-DETR, YOLO)
│   ├── inference/                    # Inference and COCO evaluation pipeline
│   └── fine_tune/
│       ├── fine_tune_detr.py         # DETR fine-tuning (+ DeART adaptation)
│       ├── fine_tune_faster_rcnn.py  # Faster R-CNN fine-tuning
│       ├── fine_tune_rt_detr.py      # RT-DETR fine-tuning
│       ├── fine_tune_yolo.py         # YOLO fine-tuning
│       ├── utils.py                  # Shared training utilities
│       ├── configs/
│       │   ├── single_run/           # Per-model YAML configs for individual runs
│       │   └── sweeps/               # W&B sweep YAML configs
│       └── plots_for_slides/         # Quantitative and qualitative results for the presentation
│           ├── analyze_sweep_results.ipynb
│           ├── NMS_IoU_optimization.ipynb
│           ├── qualitative_evaluation_*.py
│           └── qualitative_deart.py
└── scripts/
    ├── train/                        # SLURM training job scripts
    ├── sweep/                        # SLURM W&B sweep scripts
    └── eval/                         # SLURM evaluation and inference scripts
```

---

## Running Experiments

All commands must be run from the **`Week1/`** directory.

### Single training run

Each model has its own fine-tuning script and a corresponding config file. For example, to fine-tune DETR:

```bash
cd Week1/
python src/fine_tune/fine_tune_detr.py --config src/fine_tune/configs/single_run/config_detr.yaml
```

All other models follow the same pattern — just swap the script and config:

| Model | Script | Config |
|---|---|---|
| DETR | `fine_tune_detr.py` | `config_detr.yaml` |
| Faster R-CNN | `fine_tune_faster_rcnn.py` | `config_final_faster_rcnn.yaml` |
| RT-DETR | `fine_tune_rt_detr.py` | `config_rtdetr.yaml` |
| YOLO | `fine_tune_yolo.py` | `config_yolo_base.yaml` |

To submit as a SLURM job instead:

```bash
sbatch scripts/train/run_finetune.sh
```

### Hyperparameter sweeps (Weights & Biases)

First, register the sweep with W&B to get a sweep ID:

```bash
cd Week1/
wandb sweep src/fine_tune/configs/sweeps/sweep_lr_detr.yaml
# → copy the sweep ID printed (e.g. group01/C5/abc123)
```

Then launch the SLURM agent to run the sweep:

```bash
sbatch scripts/sweep/run_sweep.sh <SWEEP_ID> <PROJECT_NAME>
```

All sweep configs in `configs/sweeps/` work the same way.

---

## Results and Presentation Plots

Quantitative sweep analysis and qualitative visualizations used in the presentation are in:

```
Week1/src/fine_tune/plots_for_slides/
```

- `analyze_sweep_results.ipynb` — sweep comparison bar charts and domain shift analysis
- `NMS_IoU_optimization.ipynb` — NMS threshold optimization
- `qualitative_evaluation_*.py` — side-by-side GT vs prediction plots per model
- `qualitative_deart.py` — zero-shot vs LoRA fine-tuned comparison on DEArt images