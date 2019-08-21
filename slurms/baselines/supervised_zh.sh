#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/resultsctb/supervisionZHWKP%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=kann@nyu.edu;anhad@nyu.edu

# Example call: ./slurms/baselines/supervisedZH.sh 02 -1

model_no=$1
supervision_limit=$2

python -u main.py --eval_on test --treebank ctb  --force_binarize --save trained_models/baselines/supervised_zh.${model_no} --batch 25 --force_binarize --supervision_limit ${supervision_limit}
