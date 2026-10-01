#!/bin/bash
set -e
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate predformer
export LD_LIBRARY_PATH="/home/user/anaconda3/envs/predformer/lib:$LD_LIBRARY_PATH"
export PREDFORMER_MODEL=cnnstem
cd /home/user/lik/PredFormer-main

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="mmnist/${CURRENT_TIME}_PredFormer_CNNStem_stable_ep2000_bs32"

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --config_file configs/mmnist/PredFormer.py \
    --dataname mmnist \
    --data_root data \
    --res_dir work_dirs \
    --batch_size 32 \
    --epoch 2000 \
    --overwrite \
    --lr 1.2e-3 \
    --opt adamw \
    --weight_decay 1e-2 \
    --fp16 \
    --log_step 5 \
    --clip_grad 5.0 \
    --clip_mode norm \
    --ex_name "$EX_NAME" \
    --tb_dir "work_dirs/tb/${CURRENT_TIME}_stable" \
    --resume_from work_dirs/resume_stable.pth
