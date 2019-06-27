#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/uns1
python -u main.py --save trained_models/UNS_1 --batch 16 --alpha 0.0 --epochs 70 --PRPN
