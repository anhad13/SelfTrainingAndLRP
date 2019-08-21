#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/supervisionAR%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=kann@nyu.edu;anhad@nyu.edu

# Example call: ./slurms/baselines/supervised.sh 02 -1

model_no=$1
supervision_limit=$2

python -u main.py --eval_on test --force_binarize --save trained_models/baselines/supervisedAR.${model_no} --batch 64 --treebank arabic --supervision_limit ${supervision_limit}
