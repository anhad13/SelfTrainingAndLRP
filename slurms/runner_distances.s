#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/distances_1
python -u main.py --save trained_models/distances_1 --batch 64 --train_distances --parse_with_distances --alpha 1.0 --epochs 35 --PRPN
