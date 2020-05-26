Steps for self-training PRPN:

1. Training multiple PRPN instances:

python -u main.py --batch 64 --save trained_models/<model-outputpath> --alpha 1.0 --epochs 35 --PRPN --force_binarize

2. Store outputs of trained PRPN models (will be stored in the same dir as model_path):

python -u main.py --force_binarize --eval_on train --eval_only --load trained_models/${model_path} --batch 1 --PRPN

3. Generate training data from PRPN model outputs:

python -u scripts/overlap.py <comma sep outputs generated in step 2>

4. Co-train the multi-task model (from loaded pre-trained model + training outputs):

python -u main.py --batch 64 --PRPN \
    --shen --alpha 0.5 --save Fout20_12_${load_from} \
    --beta 0.5 --force_binarize --training_ratio 0.2  --load <training-data-path-from-step-1> --train_from_pickle  <training-data-path-from-step-3> --training_method interleave


