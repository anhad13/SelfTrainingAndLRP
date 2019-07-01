#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dist
#SBATCH --mail-type=END
#SBATCH --mail-user=kann@nyu.edu;anhad@nyu.edu

# Example call: ./slurms/baselines/dist.sh 02 -1

model_no=$1
supervision_limit=$2

python -u main.py --save trained_models/baselines/dist.${model_no} --batch 64 --PRPN \
    --shen --alpha 1. --parse_with_distances --beta 0. \
    --supervision_limit ${supervision_limit}
