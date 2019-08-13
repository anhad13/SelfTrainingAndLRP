#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/UNS__zh_4
python -u main.py --treebank ctb --save trained_models/UNS_zh_4 --batch 64 --alpha 0.0 --epochs 70 --PRPN --shen
