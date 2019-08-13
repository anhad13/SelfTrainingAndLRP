#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/B2B_scratch_100_0.2
#SBATCH --mail-type=END
#SBATCH --mail-user=kann@nyu.edu;anhad@nyu.edu

# Example call: ./slurms/ft_dist.sh 01 10 trained_models/baselines/semisupervised.01

supervision_limit=$1

python -u main.py --batch 64 --PRPN \
    --shen --semisupervised --training_ratio 0.2 --beta 0.5 \
    --supervision_limit ${supervision_limit} \
    --training_method "interleave"
