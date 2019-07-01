#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --output=logs/semisupervised.katha.g+d_%j
#SBATCH --mail-type=END
#SBATCH --mail-user=kann@nyu.edu

# Example call: ./slurms/semisupervised.katha/g+d.sh 02 -1

model_no=$1
supervision_limit=$2

python -u main.py --save trained_models/semisupervised.katha/g+d.${model_no} --batch 64 --PRPN \
    --shen --alpha .5 --parse_with_distances --beta 0.5 \
    --supervision_limit ${supervision_limit}
