#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/selftrain/UNS__ch_%j.out
python -u main.py --treebank ctb --dump_vocabulary --batch 64 --alpha 0.0 --epochs 70 --PRPN --shen

