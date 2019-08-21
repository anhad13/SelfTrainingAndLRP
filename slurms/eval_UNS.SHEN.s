#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/uns.shen.%j.out
#SBATCH --mail-type=END

model_no=$1

python -u main.py --treebank ctb_wkp --force_binarize --eval_on dev --eval_only --load trained_models/${model_no} --batch 1 --PRPN --shen
