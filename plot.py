"""
============================================================================
DTSKD 训练结果可视化脚本 (无需 pandas，仅依赖 matplotlib + numpy)
============================================================================
用法: python plot.py [metrics.csv路径] [输出目录]
生成: loss_and_acc.png, top1_top5.png, branches.png, lr_alpha.png, dashboard.png
============================================================================
"""

import os, sys, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ========== 中文字体设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'


def load_metrics(csv_path):
    """读取CSV指标文件"""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                data[key].append(float(val))
    for key in data:
        data[key] = np.array(data[key])

    n = len(data['epoch'])
    print(f"[Loaded] {n} epochs from {csv_path}")
    print(f"  Best val_top1: {data['val_top1'].max():.2f}% "
          f"at epoch {int(data['epoch'][data['val_top1'].argmax()])}")
    print(f"  Best val_top5: {data['val_top5'].max():.2f}% "
          f"at epoch {int(data['epoch'][data['val_top5'].argmax()])}")
    print(f"  Final (epoch {int(data['epoch'][-1])}): val_top1 = {data['val_top1'][-1]:.2f}%")
    return data


def plot_loss(data, out_dir):
    """图1: 训练损失 + 验证Top-1"""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_loss, color_acc = '#2C7BB6', '#D7191C'

    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Train Loss', color=color_loss, fontsize=13)
    l1, = ax1.plot(data['epoch'], data['train_loss'],
                   color=color_loss, alpha=0.7, linewidth=1.2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color=color_acc, fontsize=13)
    l2, = ax2.plot(data['epoch'], data['val_top1'],
                   color=color_acc, linewidth=1.5, label='Val Top-1 Acc')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    best_idx = data['val_top1'].argmax()
    best_acc = data['val_top1'].max()
    ax2.scatter(data['epoch'][best_idx], best_acc, color='red', s=60, zorder=5)
    ax2.annotate(f'Best: {best_acc:.2f}% @ Epoch {int(data["epoch"][best_idx])}',
                 xy=(data['epoch'][best_idx], best_acc),
                 xytext=(data['epoch'][best_idx] + 5, best_acc - 5),
                 fontsize=10, color='red',
                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper center', fontsize=11)
    plt.title('DTSKD: Training Loss & Validation Accuracy', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'loss_and_acc.png'))
    plt.close()
    print(f"[Saved] loss_and_acc.png")


def plot_top1_top5(data, out_dir):
    """图2: Top-1 vs Top-5 准确率"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['epoch'], data['train_top1'], color='#5DA5DA', alpha=0.5,
            linewidth=1.0, label='Train Top-1')
    ax.plot(data['epoch'], data['train_top5'], color='#FAA43A', alpha=0.5,
            linewidth=1.0, label='Train Top-5')
    ax.plot(data['epoch'], data['val_top1'], color='#60BD68', linewidth=1.8,
            label='Val Top-1')
    ax.plot(data['epoch'], data['val_top5'], color='#F17CB0', linewidth=1.8,
            label='Val Top-5')

    for col, color in [('val_top1', '#60BD68'), ('val_top5', '#F17CB0')]:
        best_idx = data[col].argmax()
        best_val = data[col].max()
        ax.scatter(data['epoch'][best_idx], best_val, color=color, s=50, zorder=5)
        ax.annotate(f'{best_val:.1f}%',
                    xy=(data['epoch'][best_idx], best_val),
                    xytext=(data['epoch'][best_idx] + 3, best_val - 3),
                    fontsize=9, color=color)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.title('DTSKD: Top-1 & Top-5 Accuracy', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'top1_top5.png'))
    plt.close()
    print(f"[Saved] top1_top5.png")


def plot_branches(data, out_dir):
    """图3: 分支准确率对比"""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#D7191C', '#FDAE61', '#ABDDA4', '#2B83BA']
    labels = ['Backbone', 'Branch-1 (shallow)', 'Branch-2 (mid)', 'Branch-3 (deep)']

    ax.plot(data['epoch'], data['val_top1'], color=colors[0], linewidth=2.0, label=labels[0])
    ax.plot(data['epoch'], data['val_b1_top1'], color=colors[1], linewidth=1.2, label=labels[1])
    ax.plot(data['epoch'], data['val_b2_top1'], color=colors[2], linewidth=1.2, label=labels[2])
    ax.plot(data['epoch'], data['val_b3_top1'], color=colors[3], linewidth=1.2, label=labels[3])

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.title('DTSKD: Multi-Branch Accuracy Comparison', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'branches.png'))
    plt.close()
    print(f"[Saved] branches.png")


def plot_lr_alpha(data, out_dir):
    """图4: 学习率 & alpha_t 衰减"""
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    color_lr, color_alpha = '#4D4D4D', '#E31A1C'

    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Learning Rate', color=color_lr, fontsize=13)
    l1, = ax1.plot(data['epoch'], data['lr'], color=color_lr, linewidth=1.5, label='Learning Rate')
    ax1.tick_params(axis='y', labelcolor=color_lr)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$\alpha_t$ (History-Teacher Weight)', color=color_alpha, fontsize=13)
    l2, = ax2.plot(data['epoch'], data['alpha_t'], color=color_alpha, linewidth=2.0,
                   label=r'$\alpha_t$')
    ax2.tick_params(axis='y', labelcolor=color_alpha)
    ax2.set_ylim(-0.05, 1.05)

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=11)
    plt.title('DTSKD: Learning Rate & Alpha Decay', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'lr_alpha.png'))
    plt.close()
    print(f"[Saved] lr_alpha.png")


def plot_dashboard(data, out_dir):
    """图5: 综合仪表盘"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 损失曲线
    ax = axes[0, 0]
    ax.plot(data['epoch'], data['train_loss'], color='#2C7BB6', linewidth=1.0, alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss'); ax.grid(True, alpha=0.3)

    # 子图2: 准确率
    ax = axes[0, 1]
    ax.plot(data['epoch'], data['train_top1'], color='#5DA5DA', alpha=0.5, linewidth=0.8,
            label='Train')
    ax.plot(data['epoch'], data['val_top1'], color='#D7191C', linewidth=1.5, label='Val')
    best_idx = data['val_top1'].argmax()
    ax.scatter(data['epoch'][best_idx], data['val_top1'].max(), color='red', s=40, zorder=5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Top-1 Acc (%)')
    ax.set_title('Top-1 Accuracy'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 子图3: 分支对比
    ax = axes[1, 0]
    ax.plot(data['epoch'], data['val_top1'], color='#D7191C', linewidth=1.5, label='Backbone')
    ax.plot(data['epoch'], data['val_b1_top1'], color='#FDAE61', linewidth=0.8, label='B1')
    ax.plot(data['epoch'], data['val_b2_top1'], color='#ABDDA4', linewidth=0.8, label='B2')
    ax.plot(data['epoch'], data['val_b3_top1'], color='#2B83BA', linewidth=0.8, label='B3')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Top-1 Acc (%)')
    ax.set_title('Branch Accuracy'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 子图4: Alpha_t 衰减
    ax = axes[1, 1]
    ax.plot(data['epoch'], data['alpha_t'], color='#E31A1C', linewidth=2.0)
    ax.fill_between(data['epoch'], 0, data['alpha_t'], alpha=0.15, color='#E31A1C')
    ax.set_xlabel('Epoch'); ax.set_ylabel(r'$\alpha_t$')
    ax.set_title(r'History-Teacher Weight $\alpha_t$'); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.suptitle('DTSKD Training Dashboard', fontsize=17, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, 'dashboard.png'))
    plt.close()
    print(f"[Saved] dashboard.png")


def main():
    if len(sys.argv) < 2:
        csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'experiments', 'dtskd_resnet18_test', 'log', 'metrics.csv'
        )
    else:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"[Error] CSV file not found: {csv_path}")
        print("Usage: python plot.py [path/to/metrics.csv] [output_dir]")
        sys.exit(1)

    out_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Info] Reading metrics from: {csv_path}")
    print(f"[Info] Output directory:    {out_dir}")
    print("-" * 50)

    data = load_metrics(csv_path)

    plot_loss(data, out_dir)
    plot_top1_top5(data, out_dir)
    plot_branches(data, out_dir)
    plot_lr_alpha(data, out_dir)
    plot_dashboard(data, out_dir)

    print("-" * 50)
    print(f"[Done] All 5 plots saved to: {out_dir}")


if __name__ == '__main__':
    main()
