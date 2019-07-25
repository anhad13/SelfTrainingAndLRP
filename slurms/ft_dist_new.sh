#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/dist
#SBATCH --mail-type=END
#SBATCH --mail-user=kann@nyu.edu;anhad@nyu.edu

# Example call: ./slurms/ft_dist.sh 01 10 trained_models/baselines/semisupervised.01

model_no=$1
supervision_limit=$2
load_from=$3

python -u main.py --save trained_models/ft/dist.${model_no} --batch 2 --PRPN \
    --shen --beta 0. \
    --supervision_limit ${supervision_limit} \
    --load $load_from \
    --training_method "interleave" \
    --batch 1
