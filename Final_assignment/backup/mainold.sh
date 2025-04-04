wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 40 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "actualBig1x1convend" \