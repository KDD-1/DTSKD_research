"""
============================================================================
SKD 遗忘检测分析脚�?============================================================================
基于 Stern et al. (AAAI 2025 / TPAMI 2026) �?Forget Fraction 度量�?分析训练过程中每�?epoch 的样本级遗忘行为�?
用法:
  python analyze_forgetting.py <experiment_dir> [--output_dir <dir>]

示例:
  python analyze_forgetting.py experiments/dtskd_full_300

输入: {experiment_dir}/log/forgetting/correct_epoch_*.pt
输出: forgetting_curves.png, forgotten_samples.csv, forgetting_report.txt
============================================================================
"""
import os, sys, csv, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'


def load_per_sample_correct(forgetting_dir):
    """加载所�?epoch �?per-sample correctness 张量

    Returns:
        correct_matrix: (num_epochs, num_samples) bool numpy array
        epochs: list of epoch numbers
    """
    files = sorted([f for f in os.listdir(forgetting_dir) if f.startswith('correct_epoch_')])
    if not files:
        raise FileNotFoundError(f"No correct_epoch_*.pt files found in {forgetting_dir}")

    # 加载第一个文件确定尺�?    first = torch.load(os.path.join(forgetting_dir, files[0]))
    n_samples = len(first)
    n_epochs = len(files)

    correct_matrix = np.zeros((n_epochs, n_samples), dtype=bool)
    epochs = []

    for i, fname in enumerate(files):
        epoch = int(fname.replace('correct_epoch_', '').replace('.pt', ''))
        epochs.append(epoch)
        data = torch.load(os.path.join(forgetting_dir, fname))
        correct_matrix[i] = data.numpy()

    return correct_matrix, np.array(epochs)


def compute_forget_learn(correct_matrix, epochs):
    """计算每个 epoch �?Forget Fraction �?Learn Fraction

    Args:
        correct_matrix: (n_epochs, n_samples) �?每个样本在每�?epoch 是否被正确分�?        epochs: (n_epochs,) �?epoch 编号

    Returns:
        F: (n_epochs,) Forget Fraction
        L: (n_epochs,) Learn Fraction
        acc: (n_epochs,) 每个 epoch 的验证精�?        final_wrong_mask: (n_samples,) 最终模型分错的样本
        per_sample_forget_count: (n_samples,) 每个样本被遗忘的次数
    """
    n_epochs, n_samples = correct_matrix.shape
    final_epoch_idx = n_epochs - 1

    # 每个 epoch 的全局精度
    acc = correct_matrix.mean(axis=1)

    # 最终模型错误分类的样本�?M_E
    final_correct = correct_matrix[final_epoch_idx]
    final_wrong_mask = ~final_correct
    n_wrong_final = final_wrong_mask.sum()

    # 计算 Forget Fraction F_e
    # F_e = (�?epoch e 正确 �?最终错�?的样本数) / 总样本数
    F = np.zeros(n_epochs)
    L = np.zeros(n_epochs)

    for e in range(n_epochs):
        # 遗忘: epoch e 正确 �?最终错�?        forgotten = correct_matrix[e] & final_wrong_mask
        F[e] = forgotten.sum() / n_samples

        # 学习: epoch e 错误 �?最终正�?        learned = (~correct_matrix[e]) & final_correct
        L[e] = learned.sum() / n_samples

    # 每个样本被遗忘的次数（在多少�?epoch 正确但最终错误）
    per_sample_forget_count = np.zeros(n_samples, dtype=int)
    for s in range(n_samples):
        if final_wrong_mask[s]:
            per_sample_forget_count[s] = correct_matrix[:, s].sum()

    return F, L, acc, final_wrong_mask, per_sample_forget_count


def load_metrics_csv(log_dir):
    """加载 metrics.csv 获取 val_top1 �?alpha_t"""
    csv_path = os.path.join(log_dir, 'metrics.csv')
    if not os.path.exists(csv_path):
        return None, None, None

    epochs = []
    val_top1 = []
    alpha_t = []

    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
        # Check if first line is a header or data
        if first_line.startswith('epoch'):
            # Has header �?use DictReader
            f.seek(0)
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epochs.append(int(row['epoch']))
                    val_top1.append(float(row['val_top1']))
                    alpha_t.append(float(row.get('alpha_t', -1)))
                except (KeyError, ValueError):
                    continue
        else:
            # No header �?parse as positional CSV
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 7:
                    try:
                        epochs.append(int(parts[0]))
                        val_top1.append(float(parts[6]))
                        alpha_t.append(float(parts[2]) if len(parts) > 2 else -1.0)
                    except (ValueError, IndexError):
                        continue

    if not epochs:
        return None, None, None
    return np.array(epochs), np.array(val_top1), np.array(alpha_t)


def identify_forgotten_samples(correct_matrix, final_wrong_mask, n_top=50):
    """识别被遗忘最严重的样�?
    Returns:
        list of (sample_idx, forget_count, first_correct_epoch, last_correct_epoch)
    """
    n_epochs, n_samples = correct_matrix.shape
    forgotten_samples = []

    for s in range(n_samples):
        if final_wrong_mask[s]:
            correct_epochs = np.where(correct_matrix[:, s])[0]
            if len(correct_epochs) > 0:
                forgotten_samples.append({
                    'idx': s,
                    'forget_count': len(correct_epochs),
                    'first_correct': correct_epochs[0],
                    'last_correct': correct_epochs[-1],
                    'correct_span': correct_epochs[-1] - correct_epochs[0] if len(correct_epochs) > 1 else 0
                })

    forgotten_samples.sort(key=lambda x: x['forget_count'], reverse=True)
    return forgotten_samples[:n_top]


def compute_forgetting_rate(F, epochs):
    """计算遗忘速率（F 的离散导数）"""
    dF = np.diff(F) / np.diff(epochs)
    return dF


def plot_forgetting_curves(F, L, acc, epochs, metrics_epochs, val_top1, alpha_t, output_dir):
    """生成综合分析�?""

    # �?: Forget/Learn + Val Acc + Alpha
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 上图: Forget Fraction vs Learn Fraction
    ax1.plot(epochs, F * 100, 'r-', linewidth=1.5, label='Forget Fraction F_e (%)')
    ax1.plot(epochs, L * 100, 'b-', linewidth=1.5, label='Learn Fraction L_e (%)')
    ax1.plot(epochs, (F + L) * 100, 'gray', linewidth=0.8, linestyle='--', label='F_e + L_e (Churn)')
    ax1.set_ylabel('Fraction of Val Set (%)', fontsize=12)
    ax1.set_title('Forgetting & Learning Dynamics During Training', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 标记最大遗忘的 epoch
    max_F_idx = np.argmax(F)
    ax1.axvline(x=epochs[max_F_idx], color='red', linestyle=':', alpha=0.5)
    ax1.annotate(f'Max Forget: epoch {epochs[max_F_idx]}\nF={F[max_F_idx]*100:.2f}%',
                 xy=(epochs[max_F_idx], F[max_F_idx]*100),
                 xytext=(epochs[max_F_idx]+20, F[max_F_idx]*100+0.5),
                 arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)

    # 下图: Val Accuracy + Alpha
    if metrics_epochs is not None:
        ax2.plot(metrics_epochs, val_top1, 'g-', linewidth=1.5, label='Val Top-1 (%)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Val Top-1 Accuracy (%)', fontsize=12, color='green')

    if alpha_t is not None:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(metrics_epochs, alpha_t, 'orange', linewidth=1, linestyle='--', label='α_t')
        ax2_twin.set_ylabel('α_t', fontsize=12, color='orange')
        ax2_twin.legend(loc='upper right', fontsize=10)

    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'forgetting_curves.png'))
    plt.close()
    print(f"[Saved] forgetting_curves.png")

    # �?: Forgetting Rate + α_t 相关�?    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    dF = compute_forgetting_rate(F, epochs)
    ax1.plot(epochs[1:], dF * 100, 'r-', linewidth=1)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Forgetting Rate (ΔF/Δepoch, %)')
    ax1.set_title('Forgetting Rate Over Training')
    ax1.grid(True, alpha=0.3)

    # F vs alpha_t scatter
    if alpha_t is not None and metrics_epochs is not None:
        # 对齐 epoch
        common_epochs = np.intersect1d(epochs.astype(int), metrics_epochs.astype(int))
        F_aligned = np.array([F[np.where(epochs == e)[0][0]] for e in common_epochs])
        alpha_aligned = np.array([alpha_t[np.where(metrics_epochs == e)[0][0]] for e in common_epochs])

        ax2.scatter(alpha_aligned, F_aligned * 100, c=common_epochs, cmap='viridis', alpha=0.6, s=20)
        ax2.set_xlabel('α_t')
        ax2.set_ylabel('Forget Fraction F_e (%)')
        ax2.set_title('F_e vs α_t (colored by epoch)')
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Epoch')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'forgetting_rate.png'))
    plt.close()
    print(f"[Saved] forgetting_rate.png")

    # �?: Per-class forgetting
    # Note: 需要加载验证集标签才能按类别分�?    # 这里生成占位�?

def compute_class_forgetting(correct_matrix, val_targets, num_classes=100):
    """计算每个类别的遗忘率"""
    n_epochs, n_samples = correct_matrix.shape
    final_correct = correct_matrix[-1]

    class_forget_frac = np.zeros(num_classes)
    class_n_samples = np.zeros(num_classes)

    for c in range(num_classes):
        class_mask = (val_targets == c)
        class_n_samples[c] = class_mask.sum()
        if class_n_samples[c] > 0:
            wrong_final_in_class = (~final_correct) & class_mask
            n_wrong = wrong_final_in_class.sum()
            if n_wrong > 0:
                # 该类中被遗忘的样本在"曾正�?的比�?                forgotten_in_class = correct_matrix[:, wrong_final_in_class].any(axis=0).sum()
                class_forget_frac[c] = forgotten_in_class / class_n_samples[c]

    return class_forget_frac, class_n_samples


def generate_report(F, L, acc, epochs, forgotten_samples, dF, output_dir):
    """生成文本报告"""
    report_path = os.path.join(output_dir, 'forgetting_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("SKD 遗忘检测分析报告\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"训练总轮�? {len(epochs)}\n")
        f.write(f"验证集样本数: {len(acc)}\n\n")

        f.write(f"--- 全局指标 ---\n")
        f.write(f"初始验证精度 (epoch {epochs[0]}): {acc[0]*100:.2f}%\n")
        f.write(f"最终验证精�?(epoch {epochs[-1]}): {acc[-1]*100:.2f}%\n")
        f.write(f"最高验证精�? {acc.max()*100:.2f}% (epoch {epochs[np.argmax(acc)]})\n\n")

        f.write(f"--- 遗忘指标 ---\n")
        f.write(f"最�?Forget Fraction: {F.max()*100:.2f}% (epoch {epochs[np.argmax(F)]})\n")
        f.write(f"最�?Forget Fraction: {F[-1]*100:.2f}%\n")
        f.write(f"最�?Learn Fraction: {L[-1]*100:.2f}%\n")
        f.write(f"净遗忘 (F-L at final epoch): {(F[-1]-L[-1])*100:.2f}%\n")
        f.write(f"�?Churn (F+L at final epoch): {(F[-1]+L[-1])*100:.2f}%\n\n")

        f.write(f"--- 遗忘动力�?---\n")
        f.write(f"遗忘速率最大�? {dF.max()*100:.4f}%/epoch (epoch {epochs[1:][np.argmax(dF)]})\n")

        # 找出遗忘加速的转折�?        mid_point = len(dF) // 2
        early_rate = np.mean(dF[:mid_point])
        late_rate = np.mean(dF[mid_point:])
        f.write(f"早期遗忘速率 (�?0%): {early_rate*100:.4f}%/epoch\n")
        f.write(f"后期遗忘速率 (�?0%): {late_rate*100:.4f}%/epoch\n")
        if late_rate > early_rate:
            f.write("�?遗忘在后期加速！可能对应 α_t 衰减后的自训练效应\n")
        else:
            f.write("遗忘速率在后期放缓\n")

        f.write(f"\n--- 被遗忘最严重�?Top-20 样本 ---\n")
        f.write(f"{'Rank':<6}{'SampleIdx':<12}{'ForgetCount':<14}{'FirstCorrect':<14}{'LastCorrect':<14}{'Span':<8}\n")
        for i, s in enumerate(forgotten_samples[:20]):
            f.write(f"{i+1:<6}{s['idx']:<12}{s['forget_count']:<14}"
                    f"{s['first_correct']:<14}{s['last_correct']:<14}{s['correct_span']:<8}\n")

    print(f"[Saved] {report_path}")


def main():
    parser = argparse.ArgumentParser(description='SKD Forgetting Analysis')
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for plots')
    parser.add_argument('--val_targets_path', type=str, default=None,
                        help='Path to val_targets.pt (generated if not provided)')
    args = parser.parse_args()

    # 路径设置
    exp_dir = args.experiment_dir
    forgetting_dir = os.path.join(exp_dir, 'log', 'forgetting')
    log_dir = os.path.join(exp_dir, 'log')
    output_dir = args.output_dir or os.path.join(exp_dir, 'log')

    if not os.path.exists(forgetting_dir):
        print(f"ERROR: forgetting directory not found: {forgetting_dir}")
        print("Make sure you ran training with --track_forgetting 1")
        sys.exit(1)

    # 1. 加载 per-sample correctness
    print("[1/5] Loading per-sample correctness data...")
    correct_matrix, epochs = load_per_sample_correct(forgetting_dir)
    print(f"  Loaded {len(epochs)} epochs × {correct_matrix.shape[1]} samples")

    # 2. 计算 Forget/Learn 分数
    print("[2/5] Computing Forget & Learn fractions...")
    F, L, acc, final_wrong_mask, per_sample_forget_count = compute_forget_learn(correct_matrix, epochs)
    print(f"  Final accuracy: {acc[-1]*100:.2f}%")
    print(f"  Max Forget: {F.max()*100:.2f}% at epoch {epochs[np.argmax(F)]}")
    print(f"  Max Learn: {L.max()*100:.2f}% at epoch {epochs[np.argmax(L)]}")

    # 3. 识别被遗忘样�?    print("[3/5] Identifying most-forgotten samples...")
    forgotten_samples = identify_forgotten_samples(correct_matrix, final_wrong_mask)
    print(f"  Total forgotten samples (final wrong): {final_wrong_mask.sum()}")
    print(f"  Top-5 most forgotten: {[(s['idx'], s['forget_count']) for s in forgotten_samples[:5]]}")

    # 4. 加载 metrics CSV
    print("[4/5] Loading metrics CSV...")
    metrics_epochs, val_top1, alpha_t = load_metrics_csv(log_dir)
    if metrics_epochs is not None:
        print(f"  Loaded {len(metrics_epochs)} epochs of metrics")

    # 5. 生成图表和报�?    print("[5/5] Generating plots and report...")
    dF = compute_forgetting_rate(F, epochs)

    plot_forgetting_curves(F, L, acc, epochs, metrics_epochs, val_top1, alpha_t, output_dir)
    generate_report(F, L, acc, epochs, forgotten_samples, dF, output_dir)

    # 保存被遗忘样本列�?    forgotten_csv = os.path.join(output_dir, 'forgotten_samples.csv')
    with open(forgotten_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'sample_idx', 'forget_count', 'first_correct_epoch', 'last_correct_epoch', 'correct_span'])
        for i, s in enumerate(forgotten_samples):
            writer.writerow([i+1, s['idx'], s['forget_count'], s['first_correct'], s['last_correct'], s['correct_span']])
    print(f"[Saved] {forgotten_csv}")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
