wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 2 \
    --epochs 100 \
    --lr 0.0005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "enorm_512_gauss_0005_bs128" \