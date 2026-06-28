#!/bin/bash
# ============================================================================
# OpenI 启智平台 — DTSKD 训练启动脚本
# 在云脑训练任务中直接运行此脚本，或复制下面命令到"执行命令"框
# ============================================================================

# OpenI 环境变量（平台自动注入）
# CODE_DIR=/code/DTSKD     # 你的代码目录
# DATASET_DIR=/dataset     # 挂载的数据集目录
# OUTPUT_DIR=/model        # 模型输出目录

set -e

echo "=== DTSKD Training on OpenI ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# --------------------------------------------------
# 1. 噪声实验 (快速跑通)
# --------------------------------------------------
# 50% 噪声 HistKD (200 epoch, ~40min on 3090)
python main.py \
    --experiments_name noise50_hist_s27 \
    --experiments_dir ./experiments \
    --HSKD 1 \
    --ce_weight 1.0 \
    --kd_weight 0.0 \
    --end_epoch 200 \
    --batch_size 128 \
    --data_type cifar100 \
    --classifier_type resnet18_dtskd \
    --data_path ./data \
    --workers 4 \
    --track_forgetting 1 \
    --random_seed 27 \
    --lr 0.1 \
    --lr_decay_schedule 100 150 \
    --coeff_decay cos \
    --cos_max 0.9 \
    --cos_min 0.0 \
    --alpha_end_epoch 200 \
    --noise_rate 0.5

echo ""
echo "=== Done ==="
