#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/supervisionZHwkp%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=kann@nyu.edu;anhad@nyu.edu

# Example call: ./slurms/baselines/supervised.sh 02 -1

model_no=$1
supervision_limit=$2

python -u main.py --eval_on dev --eval_only --force_binarize --load trained_models/${model_no} --batch 64 --treebank ctb_wkp --supervision_limit ${supervision_limit}
