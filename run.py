#!/usr/bin/env python
#coding=utf-8
"""
OpenI 云脑训练启动器 — 绕过参数传递问题，直接硬编码实验配置
"""

import os
import sys

# OpenI 环境：确保当前目录在 Python path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 硬编码实验参数
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
    '--data_path', './data',
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

# 直接运行 main.py
with open('main.py', 'r', encoding='utf-8') as f:
    code = compile(f.read(), 'main.py', 'exec')
    exec(code, {'__name__': '__main__'})
