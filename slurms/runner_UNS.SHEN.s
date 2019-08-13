#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/uns.shen.%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=kann@nyu.edu

model_no=$1

python -u main.py --save trained_models/UNS_${model_no} --batch 64 --epochs 30 --PRPN --shen
