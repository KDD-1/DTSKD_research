#!/usr/bin/env python
#coding=utf-8
"""
OpenI 云脑训练启动器 v4
- c2net 获取 CIFAR-100 数据集路径
- 适配 torchvision 目录结构
- 日志写入文件
- 训练完成后 upload_output()
"""

import os, sys, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 1. 初始化 c2net + 重定向日志
# ============================================================
from c2net.context import prepare, upload_output

c2net_context = prepare()
output_dir = c2net_context.output_path
dataset_base = c2net_context.dataset_path

# 重定向输出到文件
log_path = os.path.join(output_dir, 'debug_output.txt')
log_file = open(log_path, 'w', encoding='utf-8')

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files:
            f.write(s)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

print("=" * 60)
print("DTSKD OpenI Launcher v4")
print("=" * 60)
print(f"CWD: {os.getcwd()}")
print(f"Python: {sys.version}")
print(f"dataset_path: {dataset_base}")
print(f"output_path: {output_dir}")

# ============================================================
# 2. 适配 CIFAR-100 数据集
# ============================================================
cifar_src = os.path.join(dataset_base, "CIFAR-100")
print(f"\nCIFAR-100 source: {cifar_src}")
print(f"Source exists: {os.path.exists(cifar_src)}")
if os.path.exists(cifar_src):
    print(f"Source contents: {os.listdir(cifar_src)}")

# torchvision 期望: root/cifar-100-python/ 下有 train 和 test 文件
data_root = "./data"
target = os.path.join(data_root, "cifar-100-python")
os.makedirs(data_root, exist_ok=True)

# 尝试各种 OpenI 可能的数据集结构
found = False
for candidate in [
    cifar_src,                                    # 本身就是 cifar-100-python
    os.path.join(cifar_src, "cifar-100-python"),  # 包了一层
    os.path.join(cifar_src, "CIFAR-100"),         # 又包了一层
]:
    if os.path.isdir(candidate):
        inner = os.listdir(candidate)
        print(f"Checking {candidate}: {inner[:5]}...")
        if "train" in inner and "test" in inner:
            # 找到正确结构
            if not os.path.exists(target):
                os.symlink(candidate, target)
                print(f"  -> Linked {candidate} -> {target}")
            found = True
            break

if not found:
    print("WARNING: Could not find cifar-100-python structure!")
    print("Trying: copy everything into ./data/cifar-100-python/")
    os.makedirs(target, exist_ok=True)
    # 尝试从 cifar_src 递归复制可能的 pickle 文件
    import glob as _glob
    for root, dirs, files in os.walk(cifar_src):
        for f in files:
            if f in ('train', 'test', 'meta'):
                src = os.path.join(root, f)
                dst = os.path.join(target, f)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"  Copied {f} to {target}")
    if os.path.exists(os.path.join(target, 'train')):
        found = True

print(f"\nFinal data root: {data_root}")
print(f"Target exists: {os.path.exists(target)}")
if os.path.exists(target):
    print(f"Target contents: {os.listdir(target)}")

# 如果都失败了，尝试 torchvision 下载
if not found:
    print("Trying internet download via torchvision...")
    try:
        import urllib.request
        urllib.request.urlopen("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", timeout=10)
        print("Internet OK, torchvision should auto-download")
    except Exception as e:
        print(f"Internet NOT available: {e}")
        print("Proceeding anyway (will fail if data missing)...")

# ============================================================
# 3. 配置实验参数
# ============================================================
sys.argv = [
    'main.py',
    '--experiments_name', 'noise50_hist_s27',
    '--experiments_dir', output_dir,        # 输出到 OpenI output 目录
    '--HSKD', '1',
    '--ce_weight', '1.0',
    '--kd_weight', '0.0',
    '--end_epoch', '200',
    '--batch_size', '128',
    '--data_type', 'cifar100',
    '--classifier_type', 'resnet18_dtskd',
    '--data_path', data_root,
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

print(f"\n{'=' * 60}")
print(f"Args: {' '.join(sys.argv[1:])}")
print(f"{'=' * 60}\n")
sys.stdout.flush()

# ============================================================
# 4. 运行训练
# ============================================================
try:
    with open('main.py', 'r', encoding='utf-8') as f:
        code = compile(f.read(), 'main.py', 'exec')
        exec(code, {'__name__': '__main__'})
except Exception as e:
    print(f"\nFATAL: {e}")
    import traceback; traceback.print_exc()

# ============================================================
# 5. 上传结果
# ============================================================
print("\n=== Uploading results... ===")
upload_output()
print("=== Done ===")
log_file.close()
