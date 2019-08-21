#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/B2B_eval%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=kann@nyu.edu;anhad@nyu.edu

# Example call: ./slurms/ft_dist.sh 01 10 trained_models/baselines/semisupervised.01

supervision_limit=$1
load_from=$2

python -u main.py --batch 64 --PRPN \
    --shen --parse_with_distances --semisupervised --eval_only --eval_on test --force_binarize --beta 0.5 \
    --supervision_limit ${supervision_limit} \
    --load ${load_from} \
    --training_method "interleave"
