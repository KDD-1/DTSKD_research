"""Anti-Forgetting SKD 超参数网格搜�?搜索 α (尺度因子), τ (中性阈�?, λ (约束强度) 的最优组�?每个组合�?20 epoch, 快速筛选最佳参�?"""
import subprocess
import os
import sys
import json
from datetime import datetime

VENV_PYTHON = r"D:\Python_code\DTSKD\venv\Scripts\python.exe"
WORK_DIR = r"D:\Python_code\DTSKD"

# 固定参数
BASE_ARGS = [
    "--experiments_dir", "./experiments",
    "--HSKD", "1",
    "--ce_weight", "1.0",
    "--kd_weight", "0.0",
    "--end_epoch", "20",
    "--batch_size", "64",
    "--data_type", "cifar100",
    "--classifier_type", "resnet18_dtskd",
    "--data_path", "./data",
    "--workers", "0",
    "--track_forgetting", "1",
    "--random_seed", "27",
    "--lr", "0.1",
    "--lr_decay_schedule", "100", "150",  # 20 epochs won't hit decay
    "--coeff_decay", "cos",
    "--cos_max", "0.9",
    "--cos_min", "0.0",
    "--alpha_end_epoch", "200",
]

# 网格: 7 个组�?(默认 + 3 个参数各 2 个变�?
GRID = [
    # (name_suffix, af_alpha, af_tau, af_lambda)
    ("default",     2.0, 0.2, 0.5),   # 默认
    ("alpha_1.0",   1.0, 0.2, 0.5),   # α �? 温和权重
    ("alpha_4.0",   4.0, 0.2, 0.5),   # α �? 激进保�?    ("tau_0.1",     2.0, 0.1, 0.5),   # τ �? 易提�?    ("tau_0.4",     2.0, 0.4, 0.5),   # τ �? 严格门槛
    ("lambda_0.1",  2.0, 0.2, 0.1),   # λ �? 弱约�?    ("lambda_1.0",  2.0, 0.2, 1.0),   # λ �? 强约�?]

results = []

for name_suffix, alpha, tau, lam in GRID:
    exp_name = f"grid_af_{name_suffix}"
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {exp_name}")
    print(f"  α={alpha}, τ={tau}, λ={lam}")
    print(f"{'='*60}")

    cmd = [VENV_PYTHON, "main.py",
           "--experiments_name", exp_name,
           "--af_alpha", str(alpha),
           "--af_tau", str(tau),
           "--af_lambda", str(lam)] + BASE_ARGS

    try:
        result = subprocess.run(cmd, cwd=WORK_DIR, capture_output=True, text=True, timeout=3600)
        # 解析最后几行获取验�?acc
        output_lines = result.stdout.split('\n')
        val_acc = None
        for line in reversed(output_lines):
            # 查找 "Best Acc" 或验证精度行
            if 'val_top1' in line.lower() or 'val acc' in line.lower():
                val_acc = line.strip()
                break
        # 从日志文件读取最�?acc
        log_dir = os.path.join(WORK_DIR, "experiments", exp_name, "log")
        train_log = os.path.join(log_dir, "train.log")
        best_acc = None
        if os.path.exists(train_log):
            with open(train_log) as f:
                for line in f:
                    if 'best acc' in line.lower() or 'Best Acc' in line:
                        best_acc = line.strip()

        # 读取 CSV 最后一�?        csv_path = os.path.join(WORK_DIR, "experiments", exp_name, "metrics.csv")
        last_csv = None
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_csv = lines[-1].strip()

        status = "OK" if result.returncode == 0 else f"ERR({result.returncode})"
        result_entry = {
            'name': exp_name, 'alpha': alpha, 'tau': tau, 'lambda': lam,
            'status': status, 'best_acc_log': best_acc, 'last_csv': last_csv
        }
        results.append(result_entry)
        print(f"  Status: {status}")
        if best_acc:
            print(f"  Best: {best_acc}")
        if last_csv:
            print(f"  Last CSV: {last_csv}")

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT!")
        results.append({'name': exp_name, 'alpha': alpha, 'tau': tau, 'lambda': lam,
                        'status': 'TIMEOUT', 'best_acc_log': None, 'last_csv': None})
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({'name': exp_name, 'alpha': alpha, 'tau': tau, 'lambda': lam,
                        'status': f'ERROR:{e}', 'best_acc_log': None, 'last_csv': None})

# 打印汇�?print(f"\n\n{'='*60}")
print("GRID SEARCH RESULTS")
print(f"{'='*60}")
print(f"{'Name':<25} {'α':>6} {'τ':>6} {'λ':>6} {'Status':>8} {'Best Acc'}")
print("-"*70)
for r in results:
    best = r['best_acc_log'] or 'N/A'
    print(f"{r['name']:<25} {r['alpha']:>6.1f} {r['tau']:>6.1f} {r['lambda']:>6.1f} {r['status']:>8} {best}")

# 保存 JSON
json_path = os.path.join(WORK_DIR, "grid_search_results.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to: {json_path}")
