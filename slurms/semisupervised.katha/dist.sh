#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --output=logs/dist
#SBATCH --mail-type=END
#SBATCH --mail-user=kann@nyu.edu

# Example call: ./slurms/semisupervised.katha/dist.sh 02 -1

model_no=$1
supervision_limit=$2

python -u main.py --save trained_models/semisupervised.katha/dist.${model_no} --batch 64 --PRPN \
    --shen --alpha .5 --parse_with_distances --beta 0. \
    --supervision_limit ${supervision_limit}
