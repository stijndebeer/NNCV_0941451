wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 2 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "enorm_512_biggauss_001" \