#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/UNS__ZH_WKP_UNS_%j.out
model_no=$1
python -u main.py --treebank ctb_wkp --save trained_models/UNS__ZH_WKP_${model_no} --batch 64 --alpha 0.0 --epochs 70 --PRPN --shen

