#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/gates_1
python -u main.py --save trained_models/gates_1 --batch 64 --alpha 1.0 --epochs 35 --PRPN
