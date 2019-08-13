#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/UNS__b0
python -u main.py --bagging --train_from_pickle bag0/0.7_model_0 --save trained_models/UNS__b0 --batch 64 --alpha 0.0 --epochs 70 --PRPN
