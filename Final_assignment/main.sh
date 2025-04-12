wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 50 \
    --lr 0.0007 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "bignb_dice_final" \