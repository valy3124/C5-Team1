#!/bin/bash
cd /ghome/group01/C5/vali/C5-Team1/Week4/src
echo "Running vit-gpt2"
conda run -n c5 python evaluate_pretrained.py --model_type vit-gpt2 --mode search > run_vit-gpt2.log 2>&1
echo "Running vit-bert"
conda run -n c5 python evaluate_pretrained.py --model_type vit-bert --mode search > run_vit-bert.log 2>&1
echo "Running blip"
conda run -n c5 python evaluate_pretrained.py --model_type blip --mode search > run_blip.log 2>&1
echo "Done"
