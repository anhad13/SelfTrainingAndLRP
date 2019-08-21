#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/selftrain/UNS__ch_%j.out
model_no=$1
python -u main.py --treebank ctb --save trained_models/selftrain/UNS__ch_${model_no} --batch 64 --alpha 0.0 --epochs 70 --PRPN --shen

