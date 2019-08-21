#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/selftrain/UNS__de_%j.out
model_no=$1
python -u main.py --treebank negra --force_binarize --eval_on test --save trained_models/UNS__de_${model_no} --batch 64 --alpha 0.0 --epochs 70 --PRPN --shen


