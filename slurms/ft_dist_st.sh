#!/bin/bash
#
#SBATCH -t48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/20M_0.2_g+d_run_11
#SBATCH --mail-type=END
#SBATCH --mail-user=kann@nyu.edu;anhad@nyu.edu

# Example call: ./slurms/ft_dist.sh 01 10 trained_models/baselines/semisupervised.01
load_from=$1

python -u main.py --batch 64 --PRPN \
    --shen --alpha 0.5 \
    --beta 0.5 --training_ratio 0.2  --load $load_from --train_from_pickle 20M_out.11 --training_method interleave
