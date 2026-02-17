#!/bin/bash

# Download and install Miniconda:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh

## After installing, close and reopen your terminal application or refresh it by running the following command:

source ~/miniconda3/bin/activate

## To initialize conda on all available shells, run the following command:

conda init --all

### Creating an environment:

#(you can choose whatever version of python you want, however I work with this version)

conda create --name c5 python==3.10.14
conda activate c5 

## Pytorch installation
## in the cluster the cuda version is the 12.1, so the you need to install a torch version that fits in this cuda version
## This is the stablest one that supports cuda 12.1

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121  

# Other packages necessaries are in the requirements.txt:

pip install -r requirements.txt