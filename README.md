# MCV C5 Project (Group 1)

This repository contains the work for the Computer Vision Master's project - Module C5.

## Members

- Diego Hernández Antón
- Oriol Juan Sabater
- Valentin Micu Hontan
- Xavier Pacheco Bach
- Benet Ramió Comas

## Quick Start
`setup.sh` automates the installation of Miniconda, creates the environment, and installs all necessary library dependencies for the entire project:

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

---

## Weekly Modules

The project is structured by weeks. Each week has its own directory and a `README.md` with specific execution scripts and structural details.

- **[Week 1: Object Detection and Fine-Tuning](./Week1/)** — Fine-tuning DETR, Faster R-CNN, RT-DETR, and YOLO models on KITTI-MOTS and DeART datasets.
- **[Week 2: Object Segmentation](./Week2/)** — We use GroundingDINO and the Segment Anything Model (SAM) for semantic segmentation, as well as SAM mask decoder fine-tuning.
