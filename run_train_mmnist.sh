#!/bin/bash
# PredFormer Moving MNIST 训练脚本 — 在 screen 中运行
set -e

# ---- 激活环境 ----
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate predformer
export LD_LIBRARY_PATH="/home/user/anaconda3/envs/predformer/lib:$LD_LIBRARY_PATH"

cd /home/user/lik/PredFormer-main

# ---- 训练配置 ----
GPU=1
BATCH_SIZE=16
EPOCHS=2000
LR=1e-3

CURRENT_TIME=$(date +"%Y-%m-%d-%H-%M")
EX_NAME="mmnist/${CURRENT_TIME}_PredFormer_TSST_ep${EPOCHS}_bs${BATCH_SIZE}"

echo "========================================"
echo "  PredFormer Training — Moving MNIST"
echo "  GPU: $GPU | Batch: $BATCH_SIZE | Epochs: $EPOCHS"
echo "  Experiment: $EX_NAME"
echo "========================================"
echo ""

CUDA_VISIBLE_DEVICES=$GPU python tools/train.py \
    --config_file configs/mmnist/PredFormer.py \
    --dataname mmnist \
    --data_root data \
    --res_dir work_dirs \
    --batch_size $BATCH_SIZE \
    --epoch $EPOCHS \
    --overwrite \
    --lr $LR \
    --opt adamw \
    --weight_decay 1e-2 \
    --ex_name "$EX_NAME" \
    --tb_dir "logs_tb/${CURRENT_TIME}"

echo ""
echo "训练完成！"
