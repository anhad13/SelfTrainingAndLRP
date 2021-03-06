#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/gates
#SBATCH --mail-type=END
#SBATCH --mail-user=kann@nyu.edu;anhad@nyu.edu

# Example call: ./slurms/baselines/gates.sh 02 -1

model_no=$1
supervision_limit=$2

python -u main.py --save trained_models/baselines/gates.${model_no} --batch 64 --PRPN \
    --shen --alpha 1. --beta 1. \
    --supervision_limit ${supervision_limit}
