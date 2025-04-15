wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \ #2 for 512x512
    --epochs 100 \
    --lr 0.0007 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "enorm_noMAP_notrainMAP512" \