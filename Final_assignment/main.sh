wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 10 \
    --lr 0.0007 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "final_citymean_1_0" \