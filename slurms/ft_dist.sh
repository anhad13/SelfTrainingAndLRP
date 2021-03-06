#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/b2b_SEMISUP_500
#SBATCH --mail-type=END
#SBATCH --mail-user=kann@nyu.edu;anhad@nyu.edu

# Example call: ./slurms/ft_dist.sh 01 10 trained_models/baselines/semisupervised.01

supervision_limit=$1
load_from=$2

python -u main.py --batch 64 --PRPN \
    --shen --alpha 1.0 \
    --beta 0.5 --training_ratio 0.1 --supervision_limit ${supervision_limit} --training_method interleave\
    --load ${load_from}
