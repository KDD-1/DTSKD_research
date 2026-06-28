"""
三合一遗忘曲线对比图
比较 CE Only / HistKD Only / Ours (MCW-AF) 的遗忘行为
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'


def load_per_sample_correct(forgetting_dir):
    files = sorted([f for f in os.listdir(forgetting_dir) if f.startswith('correct_epoch_')])
    first = torch.load(os.path.join(forgetting_dir, files[0]))
    n_samples = len(first)
    n_epochs = len(files)
    correct_matrix = np.zeros((n_epochs, n_samples), dtype=bool)
    epochs = []
    for i, fname in enumerate(files):
        epoch = int(fname.replace('correct_epoch_', '').replace('.pt', ''))
        epochs.append(epoch)
        correct_matrix[i] = torch.load(os.path.join(forgetting_dir, fname)).numpy()
    return correct_matrix, np.array(epochs)


def compute_forget_learn(correct_matrix, epochs):
    n_epochs, n_samples = correct_matrix.shape
    final_idx = n_epochs - 1
    acc = correct_matrix.mean(axis=1)
    final_correct = correct_matrix[final_idx]
    final_wrong_mask = ~final_correct
    F = np.zeros(n_epochs)
    for e in range(n_epochs):
        F[e] = (correct_matrix[e] & final_wrong_mask).sum() / n_samples
    return F, acc


experiments = {
    'CE Only': 'experiments/forgetting_ce_s27',
    'HistKD Only': 'experiments/forgetting_hist_s27',
    'Ours (MCW-AF)': 'experiments/forgetting_ours_s27',
}

colors = {'CE Only': '#E74C3C', 'HistKD Only': '#3498DB', 'Ours (MCW-AF)': '#2ECC71'}
linestyles = {'CE Only': '--', 'HistKD Only': '-', 'Ours (MCW-AF)': '-'}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# --- Plot 1: Forget Fraction ---
ax = axes[0]
for label, exp_dir in experiments.items():
    forget_dir = os.path.join(exp_dir, 'log', 'forgetting')
    cm, ep = load_per_sample_correct(forget_dir)
    F, acc = compute_forget_learn(cm, ep)
    ax.plot(ep, F * 100, color=colors[label], linestyle=linestyles[label],
            linewidth=2.0, label=label, alpha=0.9)
    max_idx = np.argmax(F)
    ax.annotate(f'{F[max_idx]*100:.2f}%',
                xy=(ep[max_idx], F[max_idx]*100),
                xytext=(ep[max_idx]+15, F[max_idx]*100+0.3),
                fontsize=8, color=colors[label], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=colors[label], alpha=0.6))

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Forget Fraction (%)', fontsize=12)
ax.set_title('Forgetting Dynamics', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

# --- Plot 2: Val Accuracy ---
ax = axes[1]
for label, exp_dir in experiments.items():
    forget_dir = os.path.join(exp_dir, 'log', 'forgetting')
    cm, ep = load_per_sample_correct(forget_dir)
    F, acc = compute_forget_learn(cm, ep)
    ax.plot(ep, acc * 100, color=colors[label], linestyle=linestyles[label],
            linewidth=2.0, label=label, alpha=0.9)
    best_idx = np.argmax(acc)
    ax.axhline(y=acc[best_idx]*100, color=colors[label], linestyle=':', alpha=0.3)
    ax.annotate(f'{acc[best_idx]*100:.2f}%',
                xy=(ep[best_idx], acc[best_idx]*100),
                fontsize=8, color=colors[label], fontweight='bold')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Val Accuracy (%)', fontsize=12)
ax.set_title('Accuracy Curves', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)

# --- Plot 3: Forget vs Acc Scatter ---
ax = axes[2]
markers = {'CE Only': 's', 'HistKD Only': 'D', 'Ours (MCW-AF)': 'o'}
for label, exp_dir in experiments.items():
    forget_dir = os.path.join(exp_dir, 'log', 'forgetting')
    cm, ep = load_per_sample_correct(forget_dir)
    F, acc = compute_forget_learn(cm, ep)
    ax.scatter(acc * 100, F * 100, c=colors[label], s=5, alpha=0.15)
    ax.scatter([acc[-1]*100], [F[-1]*100], c=colors[label], s=120, marker=markers[label],
               edgecolors='black', linewidths=1.2, zorder=5,
               label=f'{label}\nAcc={acc[-1]*100:.1f}% F={F[-1]*100:.2f}%')

ax.set_xlabel('Val Accuracy (%)', fontsize=12)
ax.set_ylabel('Forget Fraction (%)', fontsize=12)
ax.set_title('Accuracy-Forgetting Trade-off', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

plt.suptitle('DTSKD Forgetting Analysis: CE vs HistKD vs Ours (MCW-AF)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
output_path = 'experiments/forgetting_comparison.png'
plt.savefig(output_path)
plt.close()
print(f'[Saved] {output_path}')

# Summary table
print("\n" + "="*70)
print(f"{'Method':<20} {'Best Acc':>8} {'Final Acc':>8} {'Max Forget':>10} {'Final F':>8}")
print("-"*70)
for label, exp_dir in experiments.items():
    forget_dir = os.path.join(exp_dir, 'log', 'forgetting')
    cm, ep = load_per_sample_correct(forget_dir)
    F, acc = compute_forget_learn(cm, ep)
    print(f"{label:<20} {acc.max()*100:>7.2f}% {acc[-1]*100:>8.2f}% {F.max()*100:>9.2f}% {F[-1]*100:>7.2f}%")
print("="*70)
