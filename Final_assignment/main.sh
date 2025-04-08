wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 40 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "enorm_resnet_coupled_nostem" \