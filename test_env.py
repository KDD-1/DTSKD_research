#!/usr/bin/env python
#coding=utf-8
"""最小测试 — 确认 OpenI 能否运行 + 找到数据集路径"""
import os, sys

print("=== TEST START ===")
print("Python:", sys.version)
print("CWD:", os.getcwd())
print("Files:", os.listdir(".")[:10])

# 尝试用 c2net 获取数据集路径
try:
    from c2net.context import prepare
    ctx = prepare()
    print("DataSet path:", ctx.dataset_path)
    print("Output path:", ctx.output_path)
    if os.path.exists(ctx.dataset_path):
        print("DataSet contents:", os.listdir(ctx.dataset_path))
except Exception as e:
    print("c2net error:", e)

# 检查 torch
try:
    import torch
    print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
except Exception as e:
    print("torch error:", e)

print("=== TEST END ===")
