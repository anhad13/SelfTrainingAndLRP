#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/uns.shen.%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=kann@nyu.edu

model_no=$1

python -u main.py --save trained_models/UNS.shen.${model_no} --batch 16 --epochs 30 --PRPN --shen
