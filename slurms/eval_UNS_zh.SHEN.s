#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/uns.shen.%j.out
#SBATCH --mail-type=END

model_no=$1

python -u main.py --eval_on test --eval_only --treebank ctb --load trained_models/UNS_zh_${model_no} --batch 1 --PRPN --shen
