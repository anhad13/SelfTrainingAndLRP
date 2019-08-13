#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/FT_300_3
#SBATCH --mail-type=END
#SBATCH --mail-user=anhad@nyu.edu

# Example call: ./slurms/ft_dist.sh 01 10 trained_models/baselines/semisupervised.01

supervision_limit=$1
load_from=$2

python -u main.py --parse_with_distances --batch 2 --epochs 100 --PRPN \
    --shen --alpha 1.0 --beta 0.5 \
    --vocabulary dict.pkl --training_method supervised --supervision_limit ${supervision_limit} \
    --load ${load_from}
