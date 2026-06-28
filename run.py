#!/usr/bin/env python
#coding=utf-8
"""
OpenI 云脑训练启动器 — 使用 c2net 获取数据集路径
"""

import os
import sys

# 确保当前目录在 Python path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 1. 获取 OpenI 环境路径
# ============================================================
data_path = "./data"
try:
    from c2net.context import prepare
    ctx = prepare()
    # OpenI 挂载的数据集路径
    ds_path = ctx.dataset_path
    print(f"[OpenI] dataset_path = {ds_path}")
    if os.path.exists(ds_path):
        contents = os.listdir(ds_path)
        print(f"[OpenI] dataset contents: {contents}")
        # CIFAR-100 数据集挂载在这里，直接用它
        data_path = ds_path
    # 输出路径
    output_base = ctx.output_path
    print(f"[OpenI] output_path = {output_base}")
except Exception as e:
    print(f"[OpenI] c2net not available: {e}, using default ./data")

# ============================================================
# 2. 配置实验参数
# ============================================================
sys.argv = [
    'main.py',
    '--experiments_name', 'noise50_hist_s27',
    '--experiments_dir', './experiments',
    '--HSKD', '1',
    '--ce_weight', '1.0',
    '--kd_weight', '0.0',
    '--end_epoch', '200',
    '--batch_size', '128',
    '--data_type', 'cifar100',
    '--classifier_type', 'resnet18_dtskd',
    '--data_path', data_path,
    '--workers', '4',
    '--track_forgetting', '1',
    '--random_seed', '27',
    '--lr', '0.1',
    '--lr_decay_schedule', '100', '150',
    '--coeff_decay', 'cos',
    '--cos_max', '0.9',
    '--cos_min', '0.0',
    '--alpha_end_epoch', '200',
    '--noise_rate', '0.5',
]

print(f"[DTSKD] Starting with data_path={data_path}")
print(f"[DTSKD] Args: {' '.join(sys.argv[1:])}")

# ============================================================
# 3. 运行主程序
# ============================================================
with open('main.py', 'r', encoding='utf-8') as f:
    code = compile(f.read(), 'main.py', 'exec')
    exec(code, {'__name__': '__main__'})
