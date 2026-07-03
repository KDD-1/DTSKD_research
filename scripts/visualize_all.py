"""
============================================================================
DTSKD 遗忘研究 �?完整可视�?(Publication-Ready)
============================================================================
生成论文级对比图: 遗忘曲线 / 精度 / Loss / AF损失 / 分支分析 / 汇总柱状图
"""
import os, csv, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from collections import defaultdict

# ── 全局样式 ──────────────────────────────────────────────
plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# ── 配色 ──────────────────────────────────────────────────
C = {
    'ce':      '#E74C3C',  # �?    'histkd':  '#3498DB',  # �?    'ours':    '#27AE60',  # �?    'b1':      '#F39C12',  # �?    'b2':      '#9B59B6',  # �?    'b3':      '#1ABC9C',  # �?    'af_loss': '#E67E22',  # 橙红
    'lr':      '#7F8C8D',  # �?}

EXP_DIRS = {
    'CE Only':       'experiments/forgetting/forgetting_ce_s27',
    'HistKD Only':   'experiments/forgetting/forgetting_hist_s27',
    'Ours (MCW-AF)': 'experiments/forgetting/forgetting_ours_s27',
}
EXP_COLORS = {'CE Only': C['ce'], 'HistKD Only': C['histkd'], 'Ours (MCW-AF)': C['ours']}
EXP_LS =     {'CE Only': '--', 'HistKD Only': '-', 'Ours (MCW-AF)': '-'}
EXP_LW =     {'CE Only': 1.8, 'HistKD Only': 2.0, 'Ours (MCW-AF)': 2.2}

# ====================================================================
# DATA LOADING
# ====================================================================

def load_metrics(exp_dir):
    """Load metrics.csv robustly (handles with/without header, with/without af_loss)"""
    path = os.path.join(exp_dir, 'log', 'metrics.csv')
    with open(path, 'r', encoding='utf-8-sig') as f:
        first = f.readline().strip()
        f.seek(0)
        if first.startswith('epoch'):
            reader = csv.DictReader(f)
            rows = list(reader)
        else:
            cols = ['epoch','lr','alpha_t','train_loss','train_top1','train_top5',
                    'val_top1','val_top5','val_b1_top1','val_b2_top1','val_b3_top1']
            reader = csv.reader(f)
            rows = [dict(zip(cols, r)) for r in reader]
    return rows

def load_forgetting(exp_dir):
    """Load per-sample correctness matrix"""
    forget_dir = os.path.join(exp_dir, 'log', 'forgetting')
    files = sorted([f for f in os.listdir(forget_dir) if f.startswith('correct_epoch_')])
    first = torch.load(os.path.join(forget_dir, files[0]))
    n_samples, n_epochs = len(first), len(files)
    correct_matrix = np.zeros((n_epochs, n_samples), dtype=bool)
    epochs = []
    for i, fname in enumerate(files):
        epochs.append(int(fname.replace('correct_epoch_', '').replace('.pt', '')))
        correct_matrix[i] = torch.load(os.path.join(forget_dir, fname)).numpy()
    return correct_matrix, np.array(epochs)

def compute_forget_frac(correct_matrix):
    """Forget Fraction per epoch"""
    n_epochs = correct_matrix.shape[0]
    final_correct = correct_matrix[-1]
    final_wrong_mask = ~final_correct
    F = np.zeros(n_epochs)
    for e in range(n_epochs):
        F[e] = (correct_matrix[e] & final_wrong_mask).sum() / correct_matrix.shape[1]
    return F

def smooth(y, window=5):
    """Moving average smoothing"""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')

# ====================================================================
# FIGURE 1: MAIN COMPARISON (2×2)
# ====================================================================

def fig1_main_comparison():
    print("[1/5] Main comparison figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Pre-load all data
    data = {}
    for name, exp_dir in EXP_DIRS.items():
        rows = load_metrics(exp_dir)
        cm, ep = load_forgetting(exp_dir)
        F = compute_forget_frac(cm)
        acc_val = np.array([float(r['val_top1']) for r in rows])
        ep_metrics = np.array([int(r['epoch']) for r in rows])
        # Align forgetting epochs with metrics epochs
        acc_aligned = np.zeros(len(ep))
        for i, e in enumerate(ep):
            idx = np.where(ep_metrics == e)[0]
            if len(idx) > 0:
                acc_aligned[i] = acc_val[idx[0]]
            else:
                acc_aligned[i] = np.nan
        data[name] = {
            'rows': rows, 'cm': cm, 'ep': ep, 'F': F,
            'acc_val': acc_val, 'ep_metrics': ep_metrics,
            'acc_aligned': acc_aligned
        }

    # (a) Forget Fraction
    ax = axes[0, 0]
    for name in ['CE Only', 'HistKD Only', 'Ours (MCW-AF)']:
        d = data[name]
        ax.plot(d['ep'], d['F'] * 100, color=EXP_COLORS[name],
                ls=EXP_LS[name], lw=EXP_LW[name], label=name, alpha=0.9)
        # Mark max
        mi = np.argmax(d['F'])
        ax.plot(d['ep'][mi], d['F'][mi]*100, 'D', color=EXP_COLORS[name], markersize=6)
        ax.text(d['ep'][mi]+8, d['F'][mi]*100+0.08,
                f'{d["F"][mi]*100:.2f}%', fontsize=8, color=EXP_COLORS[name], fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Forget Fraction (%)')
    ax.set_title('(a) Forgetting Dynamics', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 200)

    # (b) Val Accuracy
    ax = axes[0, 1]
    for name in ['CE Only', 'HistKD Only', 'Ours (MCW-AF)']:
        d = data[name]
        ax.plot(d['ep_metrics'], d['acc_val'], color=EXP_COLORS[name],
                ls=EXP_LS[name], lw=EXP_LW[name], label=name, alpha=0.9)
        mi = np.argmax(d['acc_val'])
        ax.axhline(y=d['acc_val'][mi], color=EXP_COLORS[name], linestyle=':', alpha=0.3)
        ax.text(d['ep_metrics'][mi]+3, d['acc_val'][mi]+0.3,
                f'{d["acc_val"][mi]:.1f}%', fontsize=8, color=EXP_COLORS[name], fontweight='bold')
    # Mark LR decay points
    for ep_decay in [100, 150]:
        ax.axvline(x=ep_decay, color='gray', linestyle=':', alpha=0.4, lw=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Top-1 Accuracy (%)')
    ax.set_title('(b) Validation Accuracy', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 200)

    # (c) Train Loss
    ax = axes[1, 0]
    for name in ['CE Only', 'HistKD Only', 'Ours (MCW-AF)']:
        d = data[name]
        ep_m = d['ep_metrics']
        loss = np.array([float(r['train_loss']) for r in d['rows']])
        ax.plot(ep_m, smooth(loss, 5), color=EXP_COLORS[name],
                ls=EXP_LS[name], lw=EXP_LW[name], label=name, alpha=0.9)
        # Raw faint
        ax.plot(ep_m, loss, color=EXP_COLORS[name], alpha=0.08, lw=0.5)
    # Mark LR decay
    for ep_decay in [100, 150]:
        ax.axvline(x=ep_decay, color='gray', linestyle=':', alpha=0.4, lw=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss (smoothed)')
    ax.set_title('(c) Training Loss', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 200)
    ax.set_yscale('log')

    # (d) Summary Bar Chart
    ax = axes[1, 1]
    methods = ['CE Only', 'HistKD Only', 'Ours (MCW-AF)']
    x = np.arange(len(methods))
    width = 0.25

    best_vals = [data[m]['acc_val'].max() for m in methods]
    final_vals = [data[m]['acc_val'][-1] for m in methods]
    max_forgets = [data[m]['F'].max() * 100 for m in methods]

    bars1 = ax.bar(x - width, best_vals, width, label='Best Val Acc (%)',
                   color=[EXP_COLORS[m] for m in methods], edgecolor='white', alpha=0.85)
    bars2 = ax.bar(x, final_vals, width, label='Final Val Acc (%)',
                   color=[EXP_COLORS[m] for m in methods], edgecolor='white', alpha=0.45, hatch='///')
    # Annotate
    for bar, val in zip(bars1, best_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.3,
                f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, final_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.3,
                f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')

    # Forget Fraction on right y-axis
    ax2 = ax.twinx()
    bars3 = ax2.bar(x + width, max_forgets, width, label='Max Forget (%)',
                    color=['#E74C3C44', '#3498DB44', '#27AE6044'], edgecolor=['#E74C3C', '#3498DB', '#27AE60'],
                    linewidth=2, hatch='...')
    for bar, val in zip(bars3, max_forgets):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.05,
                 f'{val:.2f}%', ha='center', fontsize=8, fontweight='bold', color='darkred')

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(d) Summary Comparison', fontweight='bold')
    ax.legend(loc='upper left', fontsize=7.5)
    ax2.legend(loc='upper right', fontsize=7.5)
    ax.set_ylim(70, 83)
    ax2.set_ylim(0, 8)
    ax.grid(True, alpha=0.2, axis='y')

    plt.suptitle('DTSKD Forgetting Study: CE vs HistKD vs MCW-AF (Ours)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig('experiments/fig1_main_comparison.png', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print('  -> experiments/fig1_main_comparison.png')

# ====================================================================
# FIGURE 2: Ours (MCW-AF) DETAILED ANALYSIS
# ====================================================================

def fig2_ours_detail():
    print("[2/5] Ours (MCW-AF) detailed analysis...")
    rows = load_metrics('experiments/forgetting/forgetting_ours_s27')
    cm, ep = load_forgetting('experiments/forgetting/forgetting_ours_s27')
    F = compute_forget_frac(cm)

    ep_m = np.array([int(r['epoch']) for r in rows])
    af_loss = np.array([float(r.get('af_loss', 0)) for r in rows])
    acc_train = np.array([float(r['train_top1']) for r in rows])
    acc_val = np.array([float(r['val_top1']) for r in rows])
    acc_b1 = np.array([float(r['val_b1_top1']) for r in rows])
    acc_b2 = np.array([float(r['val_b2_top1']) for r in rows])
    acc_b3 = np.array([float(r['val_b3_top1']) for r in rows])
    lr = np.array([float(r['lr']) for r in rows])
    alpha_t = np.array([float(r['alpha_t']) for r in rows])

    # Align forgetting with metrics
    F_aligned = np.zeros(len(ep_m))
    for i, e in enumerate(ep_m):
        idx = np.where(ep == e)[0]
        F_aligned[i] = F[idx[0]] * 100 if len(idx) > 0 else np.nan

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) AF Loss over training
    ax = axes[0, 0]
    ax.fill_between(ep_m, 0, af_loss, color=C['af_loss'], alpha=0.15)
    ax.plot(ep_m, af_loss, color=C['af_loss'], lw=2)
    ax.plot(ep_m, smooth(af_loss, 10), color='darkred', lw=1.5, label='Smoothed')
    for ep_decay in [100, 150]:
        ax.axvline(x=ep_decay, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AF Loss')
    ax.set_title('(a) Anti-Forgetting Loss', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # (b) AF Loss vs Forget Fraction
    ax = axes[0, 1]
    # Color by epoch
    scatter = ax.scatter(F_aligned[1:], af_loss[1:], c=ep_m[1:], cmap='viridis',
                         s=15, alpha=0.7, edgecolors='none')
    ax.set_xlabel('Forget Fraction (%)')
    ax.set_ylabel('AF Loss')
    ax.set_title('(b) AF Loss vs Forget Fraction', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Epoch')
    ax.grid(True, alpha=0.25)
    # Correlation
    valid = ~np.isnan(F_aligned[1:])
    if valid.sum() > 1:
        corr = np.corrcoef(F_aligned[1:][valid], af_loss[1:][valid])[0, 1]
        ax.text(0.95, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # (c) AF Loss vs Alpha_t
    ax = axes[0, 2]
    ax.plot(ep_m, af_loss, color=C['af_loss'], lw=1.5, label='AF Loss')
    ax2 = ax.twinx()
    ax2.plot(ep_m, alpha_t, color='#8E44AD', lw=1.5, ls='--', label='α_t')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AF Loss', color=C['af_loss'])
    ax2.set_ylabel('α_t', color='#8E44AD')
    ax.set_title('(c) AF Loss & α_t Decay', fontweight='bold')
    ax.grid(True, alpha=0.25)
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # (d) Train vs Val Accuracy
    ax = axes[1, 0]
    ax.plot(ep_m, acc_train, color='#2C3E50', lw=1.5, ls='-', label='Train Top-1', alpha=0.8)
    ax.plot(ep_m, acc_val, color=C['ours'], lw=2, ls='-', label='Val Top-1', alpha=0.9)
    # Fill generalization gap
    gap = acc_train - acc_val
    ax.fill_between(ep_m, acc_val, acc_train, alpha=0.12, color='gray', label='Generalization Gap')
    for ep_decay in [100, 150]:
        ax.axvline(x=ep_decay, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(d) Train/Val Accuracy (Ours)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # (e) Branch-level Val Accuracy
    ax = axes[1, 1]
    ax.plot(ep_m, acc_b1, color=C['b1'], lw=1.5, label='Branch 1')
    ax.plot(ep_m, acc_b2, color=C['b2'], lw=1.5, label='Branch 2')
    ax.plot(ep_m, acc_b3, color=C['b3'], lw=1.5, label='Branch 3')
    ax.plot(ep_m, acc_val, color='black', lw=2.5, ls='--', label='Ensemble (Val)', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('(e) Per-Branch Performance (Ours)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # (f) Branch Gap (max - min among branches)
    ax = axes[1, 2]
    branch_stack = np.column_stack([acc_b1, acc_b2, acc_b3])
    branch_gap = branch_stack.max(axis=1) - branch_stack.min(axis=1)
    ax.fill_between(ep_m, 0, branch_gap, color=C['b2'], alpha=0.2)
    ax.plot(ep_m, branch_gap, color=C['b2'], lw=1.8)
    ax.plot(ep_m, smooth(branch_gap, 10), color='darkviolet', lw=1.5, label='Smoothed')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Branch Gap (%)')
    ax.set_title('(f) Inter-Branch Diversity (Ours)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.suptitle('MCW-AF (Ours) �?Detailed Training Dynamics', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig('experiments/fig2_ours_detail.png', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print('  -> experiments/fig2_ours_detail.png')

# ====================================================================
# FIGURE 3: FORGETTING RATE & PER-CLASS
# ====================================================================

def fig3_forgetting_rate():
    print("[3/5] Forgetting rate analysis...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, (name, exp_dir) in enumerate(EXP_DIRS.items()):
        cm, ep = load_forgetting(exp_dir)
        F = compute_forget_frac(cm)
        dF = np.diff(F) * 100  # percent per epoch

        # Forgetting Rate
        ax = axes[0]
        ax.plot(ep[1:], smooth(dF, 5), color=EXP_COLORS[name], lw=EXP_LW[name],
                ls=EXP_LS[name], label=name, alpha=0.85)
        ax.axhline(y=0, color='black', linestyle='-', lw=0.5)

        # Cumulative forgetting
        ax2 = axes[1]
        cum_F = np.cumsum(np.maximum(dF, 0))  # Only positive = actual forgetting events
        ax2.plot(ep[1:], cum_F, color=EXP_COLORS[name], lw=EXP_LW[name],
                 ls=EXP_LS[name], label=name, alpha=0.85)

        # Ratio of forgetting events
        ax3 = axes[2]
        n_pos = (dF > 0).sum()
        n_neg = (dF < 0).sum()
        ax3.bar(i, n_pos / len(dF) * 100, 0.35, color=EXP_COLORS[name], alpha=0.85,
                label='ΔF > 0 (Forgetting)' if i == 0 else '')
        ax3.bar(i, n_neg / len(dF) * 100, 0.35, bottom=n_pos / len(dF) * 100,
                color=EXP_COLORS[name], alpha=0.35, hatch='///',
                label='ΔF < 0 (Recovery)' if i == 0 else '')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Forgetting Rate (%/epoch, smoothed)')
    axes[0].set_title('(a) Forgetting Rate (dF/dE)', fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Cumulative Forgetting Events (%)')
    axes[1].set_title('(b) Cumulative Forgetting', fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)

    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(EXP_DIRS.keys())
    axes[2].set_ylabel('Fraction of Epochs (%)')
    axes[2].set_title('(c) Forgetting vs Recovery Epochs', fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.25, axis='y')

    plt.suptitle('Forgetting Dynamics: Rate, Accumulation, and Recovery', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig('experiments/fig3_forgetting_rate.png', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print('  -> experiments/fig3_forgetting_rate.png')

# ====================================================================
# FIGURE 4: LEARNING CURVES + LR SCHEDULE CONTEXT
# ====================================================================

def fig4_learning_dynamics():
    print("[4/5] Learning dynamics with LR context...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Train top-1 for all methods
    ax = axes[0, 0]
    for name in EXP_DIRS:
        rows = load_metrics(EXP_DIRS[name])
        ep_m = np.array([int(r['epoch']) for r in rows])
        acc_train = np.array([float(r['train_top1']) for r in rows])
        ax.plot(ep_m, acc_train, color=EXP_COLORS[name], ls=EXP_LS[name],
                lw=EXP_LW[name], label=name, alpha=0.85)
    for ep_decay in [100, 150]:
        ax.axvline(x=ep_decay, color='gray', linestyle=':', alpha=0.4, lw=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Top-1 (%)')
    ax.set_title('(a) Training Accuracy', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # (b) Generalization Gap (Train - Val) �?HistKD & Ours only
    ax = axes[0, 1]
    for name in ['HistKD Only', 'Ours (MCW-AF)']:
        rows = load_metrics(EXP_DIRS[name])
        ep_m = np.array([int(r['epoch']) for r in rows])
        acc_train = np.array([float(r['train_top1']) for r in rows])
        acc_val = np.array([float(r['val_top1']) for r in rows])
        gap = acc_train - acc_val
        ax.plot(ep_m, smooth(gap, 5), color=EXP_COLORS[name], ls=EXP_LS[name],
                lw=EXP_LW[name], label=name, alpha=0.85)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Generalization Gap (Train - Val, %)')
    ax.set_title('(b) Generalization Gap (Smoothed)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # (c) Val top-1 with error bands �?Ours only
    ax = axes[1, 0]
    rows = load_metrics(EXP_DIRS['Ours (MCW-AF)'])
    ep_m = np.array([int(r['epoch']) for r in rows])
    acc_val = np.array([float(r['val_top1']) for r in rows])
    acc_b1 = np.array([float(r['val_b1_top1']) for r in rows])
    acc_b2 = np.array([float(r['val_b2_top1']) for r in rows])
    acc_b3 = np.array([float(r['val_b3_top1']) for r in rows])

    branch_min = np.minimum(np.minimum(acc_b1, acc_b2), acc_b3)
    branch_max = np.maximum(np.maximum(acc_b1, acc_b2), acc_b3)
    ax.fill_between(ep_m, branch_min, branch_max, color=C['ours'], alpha=0.12)
    ax.plot(ep_m, acc_val, color=C['ours'], lw=2.5, label='Ensemble')
    ax.plot(ep_m, acc_b1, color=C['b1'], lw=0.8, alpha=0.5, label='B1')
    ax.plot(ep_m, acc_b2, color=C['b2'], lw=0.8, alpha=0.5, label='B2')
    ax.plot(ep_m, acc_b3, color=C['b3'], lw=0.8, alpha=0.5, label='B3')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('(c) Ensemble vs Branches (Ours)', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    # (d) LR schedule + alpha_t
    ax = axes[1, 1]
    rows = load_metrics(EXP_DIRS['Ours (MCW-AF)'])
    ep_m = np.array([int(r['epoch']) for r in rows])
    lr = np.array([float(r['lr']) for r in rows])
    alpha_t = np.array([float(r['alpha_t']) for r in rows])
    af_loss = np.array([float(r.get('af_loss', 0)) for r in rows])

    ax.plot(ep_m, lr, color=C['lr'], lw=2, label='Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate', color=C['lr'])
    ax.set_yscale('log')
    ax2 = ax.twinx()
    ax2.plot(ep_m, alpha_t, color='#8E44AD', lw=2, ls='--', label='α_t (KD weight)')
    ax2.set_ylabel('α_t', color='#8E44AD')
    ax.set_title('(d) LR Schedule & α_t Decay', fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center left')
    ax.grid(True, alpha=0.25)

    plt.suptitle('Training Dynamics: Accuracy, Generalization, and Schedule', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig('experiments/fig4_learning_dynamics.png', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print('  -> experiments/fig4_learning_dynamics.png')

# ====================================================================
# FIGURE 5: AF LOSS vs ACCURACY TRADE-OFF (Phase Diagram)
# ====================================================================

def fig5_af_tradeoff():
    print("[5/5] AF loss vs accuracy trade-off...")
    rows = load_metrics(EXP_DIRS['Ours (MCW-AF)'])
    ep_m = np.array([int(r['epoch']) for r in rows])
    af_loss = np.array([float(r.get('af_loss', 0)) for r in rows])
    acc_val = np.array([float(r['val_top1']) for r in rows])
    lr_val = np.array([float(r['lr']) for r in rows])

    # Three LR phases
    phase1 = lr_val == 0.1
    phase2 = lr_val == 0.01
    phase3 = lr_val == 0.001

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) AF Loss trajectory colored by LR phase
    ax = axes[0]
    ax.scatter(ep_m[phase1], af_loss[phase1], c=C['ce'], s=12, alpha=0.6, label='LR=0.1')
    ax.scatter(ep_m[phase2], af_loss[phase2], c=C['histkd'], s=12, alpha=0.6, label='LR=0.01')
    ax.scatter(ep_m[phase3], af_loss[phase3], c=C['ours'], s=12, alpha=0.6, label='LR=0.001')
    ax.plot(ep_m, smooth(af_loss, 10), color='black', lw=1.5, alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AF Loss')
    ax.set_title('(a) AF Loss by LR Phase', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # (b) Val Acc vs AF Loss
    ax = axes[1]
    ax.scatter(af_loss[phase1], acc_val[phase1], c=C['ce'], s=12, alpha=0.6, label='LR=0.1')
    ax.scatter(af_loss[phase2], acc_val[phase2], c=C['histkd'], s=12, alpha=0.6, label='LR=0.01')
    ax.scatter(af_loss[phase3], acc_val[phase3], c=C['ours'], s=12, alpha=0.6, label='LR=0.001')
    ax.set_xlabel('AF Loss')
    ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('(b) AF Loss vs Accuracy Trade-off', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.suptitle('MCW-AF: Anti-Forgetting Loss Analysis', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig('experiments/fig5_af_tradeoff.png', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print('  -> experiments/fig5_af_tradeoff.png')

# ====================================================================
# MAIN
# ====================================================================

if __name__ == '__main__':
    fig1_main_comparison()
    fig2_ours_detail()
    fig3_forgetting_rate()
    fig4_learning_dynamics()
    fig5_af_tradeoff()

    print("\n" + "="*60)
    print("All figures saved to experiments/")
    print("="*60)
    print("  fig1_main_comparison.png    �?主对比图 (遗忘+精度+Loss+汇�?")
    print("  fig2_ours_detail.png        �?Ours 详细分析 (AF/Branch/Diversity)")
    print("  fig3_forgetting_rate.png    �?遗忘速率分析")
    print("  fig4_learning_dynamics.png  �?训练动�?(Gap/Schedule)")
    print("  fig5_af_tradeoff.png        �?AF Loss 相图")
