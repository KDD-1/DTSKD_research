"""
============================================================================
20% 标签噪声实验 �?有害遗忘 vs 有益去噪分解 + 干净数据对比
============================================================================
对比: noise20_hist_s27 vs noise20_ours_s27 (MCW-AF v2)
参�? 干净数据 forgetting_hist_s27 vs forgetting_ours_s27
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

C = {'histkd': '#3498DB', 'ours': '#27AE60', 'harmful': '#E74C3C',
     'benign': '#3498DB', 'rescued': '#27AE60', 'lost': '#F39C12',
     'clean': '#95A5A6', 'noise': '#E67E22'}

# ===================================================================
# 1. 加载数据
# ===================================================================
def load_correct_matrix(exp_dir):
    forget_dir = os.path.join(exp_dir, 'log', 'forgetting')
    files = sorted([f for f in os.listdir(forget_dir) if f.startswith('correct_epoch_')])
    n_epochs = len(files) // 2 if len(files) > 100 else len(files)
    if len(files) > 100:
        files = files[:200]  # 取前200个epoch
    cm = np.zeros((len(files), 10000), dtype=bool)
    for i, f in enumerate(files):
        cm[i] = torch.load(os.path.join(forget_dir, f)).numpy()
    return cm

def load_metrics(exp_dir):
    """�?CSV 读取 val_top1 per epoch"""
    import csv
    epochs, vals = [], []
    with open(os.path.join(exp_dir, 'log', 'metrics.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            vals.append(float(row['val_top1']))
    return epochs, vals

def get_best_val(exp_dir):
    epochs, vals = load_metrics(exp_dir)
    best = max(vals)
    best_ep = epochs[vals.index(best)]
    return best, best_ep

# ===================================================================
# 2. 有害/有益遗忘分类
# ===================================================================
def classify_forgetting(cm):
    n_epochs, n_samples = cm.shape
    final_correct = cm[-1]
    wrong_mask = ~final_correct
    wrong_indices = np.where(wrong_mask)[0]

    results = []
    for idx in wrong_indices:
        trace = cm[:, idx].astype(int)

        # Stability: longest consecutive correct
        max_run = 0; cur_run = 0
        for v in trace:
            if v == 1:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0

        correct_eps = np.where(trace == 1)[0]
        if len(correct_eps) == 0:
            last_correct_ep = 0
        else:
            last_correct_ep = correct_eps[-1]

        # Oscillation: correct→wrong transitions
        transitions = sum(1 for i in range(1, len(trace))
                          if trace[i-1] == 1 and trace[i] == 0)
        max_possible = max(n_epochs // 2, 1)

        results.append({
            'idx': idx,
            'max_run': max_run,
            'last_correct': last_correct_ep,
            'total_correct': trace.sum(),
            'n_transitions': transitions,
            'stability': max_run / n_epochs,
            'late_forget': last_correct_ep / n_epochs,
            'oscillation': transitions / max_possible,
            'harmful_score': (max_run / n_epochs) * (last_correct_ep / n_epochs) * (1 - transitions / max_possible)
        })
    return results, wrong_indices


# ===================================================================
# 3. 主分�?# ===================================================================
print("=" * 70)
print("20% Label Noise �?Harmful vs Beneficial Forgetting Analysis")
print("=" * 70)

# === Noise experiments ===
print("\n[1] Loading noise20 experiments...")
cm_hist_n = load_correct_matrix('experiments/forgetting/noise/noise20_hist_s27')
cm_ours_n = load_correct_matrix('experiments/forgetting/noise/noise20_ours_s27')
print(f"  HistKD: {cm_hist_n.shape[0]} epochs × {cm_hist_n.shape[1]} samples")
print(f"  Ours:   {cm_ours_n.shape[0]} epochs × {cm_ours_n.shape[1]} samples")

hist_val_n, hist_val_ep_n = get_best_val('experiments/forgetting/noise/noise20_hist_s27')
ours_val_n, ours_val_ep_n = get_best_val('experiments/forgetting/noise/noise20_ours_s27')

hist_wrong_n = (~cm_hist_n[-1]).sum()
ours_wrong_n = (~cm_ours_n[-1]).sum()
print(f"  HistKD Best Val: {hist_val_n:.2f}% (ep{hist_val_ep_n}), Final Wrong: {hist_wrong_n}")
print(f"  Ours   Best Val: {ours_val_n:.2f}% (ep{ours_val_ep_n}), Final Wrong: {ours_wrong_n}")
print(f"  Δ Val: {ours_val_n - hist_val_n:+.2f}%")

# === Clean experiments (reference) ===
print("\n[2] Loading clean experiments (reference)...")
cm_hist_c = load_correct_matrix('experiments/forgetting/forgetting_hist_s27')
cm_ours_c = load_correct_matrix('experiments/forgetting/forgetting_ours_s27')
hist_val_c, hist_val_ep_c = get_best_val('experiments/forgetting/forgetting_hist_s27')
ours_val_c, ours_val_ep_c = get_best_val('experiments/forgetting/forgetting_ours_s27')
print(f"  Clean HistKD Best: {hist_val_c:.2f}%")
print(f"  Clean Ours   Best: {ours_val_c:.2f}%")
print(f"  Clean Δ Val: {ours_val_c - hist_val_c:+.2f}%")

# === Classify HistKD forgetting ===
print("\n[3] Classifying noise20 HistKD forgotten samples...")
hist_forgotten_n, hist_wrong_idx_n = classify_forgetting(cm_hist_n)
scores_n = np.array([s['harmful_score'] for s in hist_forgotten_n])
median_n = np.median(scores_n)
harmful_mask_n = scores_n >= median_n
benign_mask_n = ~harmful_mask_n

n_harmful = harmful_mask_n.sum()
n_benign = benign_mask_n.sum()
print(f"  Total wrong: {len(hist_forgotten_n)}")
print(f"  Harmful: {n_harmful} ({n_harmful/len(hist_forgotten_n)*100:.1f}%)")
print(f"  Benign:  {n_benign} ({n_benign/len(hist_forgotten_n)*100:.1f}%)")
print(f"  Median harmful_score: {median_n:.4f}")

# === Ours rescue analysis ===
print("\n[4] Rescue analysis...")
harmful_idx_n = np.array([s['idx'] for s, m in zip(hist_forgotten_n, harmful_mask_n) if m])
benign_idx_n = np.array([s['idx'] for s, m in zip(hist_forgotten_n, benign_mask_n) if m])

ours_final_n = cm_ours_n[-1]
rescued_harmful_n = ours_final_n[harmful_idx_n].sum()
rescued_benign_n = ours_final_n[benign_idx_n].sum()

# Lost: HistKD correct but Ours wrong
hist_final_n = cm_hist_n[-1]
ours_wrong_idx_n = np.where(~ours_final_n)[0]
lost_mask_n = hist_final_n[ours_wrong_idx_n]
n_lost_n = lost_mask_n.sum()

print(f"  Harmful rescued: {rescued_harmful_n}/{n_harmful} ({rescued_harmful_n/n_harmful*100:.1f}%)")
print(f"  Benign  rescued: {rescued_benign_n}/{n_benign} ({rescued_benign_n/n_benign*100:.1f}%)")

not_h = n_harmful - rescued_harmful_n
not_b = n_benign - rescued_benign_n
odds_n = (rescued_harmful_n / not_h) / (rescued_benign_n / not_b) if not_b > 0 and not_h > 0 else float('inf')
print(f"  Odds Ratio: {odds_n:.1f}x")
print(f"  Lost samples (HistKD correct, Ours wrong): {n_lost_n}")
print(f"  Net effect: {rescued_harmful_n + rescued_benign_n - n_lost_n:+d}")

# === Clean data reference ===
print("\n[5] Clean data reference...")
hist_forgotten_c, _ = classify_forgetting(cm_hist_c)
scores_c = np.array([s['harmful_score'] for s in hist_forgotten_c])
median_c = np.median(scores_c)
harmful_idx_c = np.array([s['idx'] for s in hist_forgotten_c if s['harmful_score'] >= median_c])
benign_idx_c = np.array([s['idx'] for s in hist_forgotten_c if s['harmful_score'] < median_c])
ours_final_c = cm_ours_c[-1]
hist_final_c = cm_hist_c[-1]

rh_c = ours_final_c[harmful_idx_c].sum()
rb_c = ours_final_c[benign_idx_c].sum()
n_hc = len(harmful_idx_c); n_bc = len(benign_idx_c)
odds_c = (rh_c / (n_hc - rh_c)) / (rb_c / (n_bc - rb_c)) if (n_bc - rb_c) > 0 else float('inf')
ours_wrong_c_idx = np.where(~ours_final_c)[0]
n_lost_c = hist_final_c[ours_wrong_c_idx].sum()

print(f"  Clean Harmful: {n_hc}, Benign: {n_bc}")
print(f"  Clean Harmful rescued: {rh_c}/{n_hc} ({rh_c/n_hc*100:.1f}%)")
print(f"  Clean Benign  rescued: {rb_c}/{n_bc} ({rb_c/n_bc*100:.1f}%)")
print(f"  Clean Odds Ratio: {odds_c:.1f}x")
print(f"  Clean Lost: {n_lost_c}")
print(f"  Clean Net: {rh_c + rb_c - n_lost_c:+d}")

# ===================================================================
# 4. Summary comparison table
# ===================================================================
print("\n" + "=" * 70)
print("COMPARISON: Clean vs 20% Noise")
print("-" * 70)
print(f"{'Metric':<35} {'Clean':>10} {'20% Noise':>10} {'Change':>10}")
print("-" * 70)
print(f"{'HistKD Best Val':<35} {hist_val_c:>9.2f}% {hist_val_n:>9.2f}% {hist_val_n-hist_val_c:>+9.2f}%")
print(f"{'Ours Best Val':<35} {ours_val_c:>9.2f}% {ours_val_n:>9.2f}% {ours_val_n-ours_val_c:>+9.2f}%")
print(f"{'Δ Val (Ours - HistKD)':<35} {ours_val_c-hist_val_c:>+9.2f}% {ours_val_n-hist_val_n:>+9.2f}% {(ours_val_n-hist_val_n)-(ours_val_c-hist_val_c):>+9.2f}%")
print(f"{'Harmful Forget Fraction':<35} {n_hc/len(hist_forgotten_c)*100:>9.1f}% {n_harmful/len(hist_forgotten_n)*100:>9.1f}% {n_harmful/len(hist_forgotten_n)*100-n_hc/len(hist_forgotten_c)*100:>+9.1f}%")
print(f"{'Harmful Rescue Rate':<35} {rh_c/n_hc*100:>9.1f}% {rescued_harmful_n/n_harmful*100:>9.1f}% {rescued_harmful_n/n_harmful*100-rh_c/n_hc*100:>+9.1f}%")
print(f"{'Benign Rescue Rate':<35} {rb_c/n_bc*100:>9.1f}% {rescued_benign_n/n_benign*100:>9.1f}% {rescued_benign_n/n_benign*100-rb_c/n_bc*100:>+9.1f}%")
print(f"{'Odds Ratio (H/B)':<35} {odds_c:>9.1f}x {odds_n:>9.1f}x {'�?:>10}")
print(f"{'Lost Samples':<35} {n_lost_c:>10} {n_lost_n:>10} {n_lost_n-n_lost_c:>+10}")
print(f"{'Net Effect':<35} {rh_c+rb_c-n_lost_c:>+10} {rescued_harmful_n+rescued_benign_n-n_lost_n:>+10} {rescued_harmful_n+rescued_benign_n-n_lost_n-(rh_c+rb_c-n_lost_c):>+10}")
print("=" * 70)

# ===================================================================
# 5. Visualization
# ===================================================================
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

# ---- (a) Val accuracy curves: Clean vs Noise ----
ax = axes[0, 0]
for exp_dir, label, color, ls in [
    ('experiments/forgetting/forgetting_hist_s27', 'HistKD clean', C['histkd'], '-'),
    ('experiments/forgetting/forgetting_ours_s27', 'Ours clean', C['ours'], '-'),
    ('experiments/forgetting/noise/noise20_hist_s27', 'HistKD 20% noise', C['histkd'], '--'),
    ('experiments/forgetting/noise/noise20_ours_s27', 'Ours 20% noise', C['ours'], '--'),
]:
    eps, vals = load_metrics(exp_dir)
    ax.plot(eps, vals, color=color, linestyle=ls, lw=1.5, label=label, alpha=0.85)

for lr_ep in [100, 150]:
    ax.axvline(x=lr_ep, color='gray', linestyle=':', lw=0.8, alpha=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Val Top-1 (%)')
ax.set_title('(a) Accuracy Curves: Clean vs 20% Noise', fontweight='bold')
ax.legend(fontsize=7, loc='lower right')

# ---- (b) Δ Accuracy (Ours - HistKD) ----
ax = axes[0, 1]
for exp_h, exp_o, label, color in [
    ('experiments/forgetting/forgetting_hist_s27', 'experiments/forgetting/forgetting_ours_s27', 'Clean', C['clean']),
    ('experiments/forgetting/noise/noise20_hist_s27', 'experiments/forgetting/noise/noise20_ours_s27', '20% Noise', C['noise']),
]:
    eps_h, vals_h = load_metrics(exp_h)
    eps_o, vals_o = load_metrics(exp_o)
    # Align by epoch
    common_eps = sorted(set(eps_h) & set(eps_o))
    delta = []
    for ep in common_eps:
        delta.append(vals_o[eps_o.index(ep)] - vals_h[eps_h.index(ep)])
    ax.plot(common_eps, delta, color=color, lw=1.8, label=label)
    ax.axhline(y=0, color='black', lw=0.5)

for lr_ep in [100, 150]:
    ax.axvline(x=lr_ep, color='gray', linestyle=':', lw=0.8, alpha=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Δ Val (Ours - HistKD) [%]')
ax.set_title('(b) Ours vs HistKD Gap', fontweight='bold')
ax.legend(fontsize=8)

# ---- (c) Harmful vs Benign distribution (noise) ----
ax = axes[0, 2]
ax.hist(scores_n, bins=40, color='#E67E22', alpha=0.6, edgecolor='black', linewidth=0.3)
ax.axvline(x=median_n, color='red', linestyle='--', lw=2,
           label=f'Median = {median_n:.3f} (n={len(hist_forgotten_n)})')
ax.set_xlabel('Harmful Score')
ax.set_ylabel('Count')
ax.set_title('(c) Harmful Score Distribution (20% Noise)', fontweight='bold')
ax.legend(fontsize=8)

# ---- (d) Rescue rates: Clean vs Noise ----
ax = axes[1, 0]
x = np.arange(4)
w = 0.35
rates = [rh_c/n_hc*100, rescued_harmful_n/n_harmful*100,
         rb_c/n_bc*100, rescued_benign_n/n_benign*100]
bar_colors = [C['harmful'], C['harmful'], C['benign'], C['benign']]
bar_alphas = [0.4, 0.85, 0.4, 0.85]
for i, (rate, color, alpha) in enumerate(zip(rates, bar_colors, bar_alphas)):
    bar = ax.bar(x[i], rate, w, color=color, alpha=alpha, edgecolor='black', linewidth=1.2)
    ax.text(x[i], rate + 0.3, f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Harmful\n(Clean)', 'Harmful\n(20% Noise)',
                    'Benign\n(Clean)', 'Benign\n(20% Noise)'])
ax.set_ylabel('Rescue Rate by Ours (%)')
ax.set_title('(d) Selective Rescue: Clean vs Noise', fontweight='bold')
ax.set_ylim(0, max(rates) + 8)

# ---- (e) Feature comparison: Harmful vs Benign (noise) ----
ax = axes[1, 1]
harmful_stats_n = [s for s in hist_forgotten_n if s['harmful_score'] >= median_n]
benign_stats_n = [s for s in hist_forgotten_n if s['harmful_score'] < median_n]
features = ['max_run', 'late_forget', 'oscillation', 'total_correct']
labels = ['Max Run\n(epochs)', 'Late Forget\n(ratio)', 'Oscillation\n(rate)', 'Total\nCorrect']
harmful_means = [np.mean([s[f] for s in harmful_stats_n]) for f in features[:3]] + \
                [np.mean([s['total_correct'] for s in harmful_stats_n])]
benign_means = [np.mean([s[f] for s in benign_stats_n]) for f in features[:3]] + \
               [np.mean([s['total_correct'] for s in benign_stats_n])]
x = np.arange(len(features))
ax.bar(x - w/2, harmful_means, w, label='Harmful', color=C['harmful'], alpha=0.85)
ax.bar(x + w/2, benign_means, w, label='Benign', color=C['benign'], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_title('(e) Feature Profiles (20% Noise)', fontweight='bold')
ax.legend(fontsize=8)

# ---- (f) Net effect comparison ----
ax = axes[1, 2]
clean_net = rh_c + rb_c - n_lost_c
noise_net = rescued_harmful_n + rescued_benign_n - n_lost_n
nets = [clean_net, noise_net]
bars = ax.bar(['Clean', '20% Noise'], nets, color=[C['clean'], C['noise']],
              alpha=0.85, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, nets):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:+d}', ha='center', fontsize=14, fontweight='bold',
            va='bottom' if val > 0 else 'top')
ax.axhline(y=0, color='black', lw=0.5)
ax.set_ylabel('Net Effect (Rescued - Lost)')
ax.set_title('(f) Net Benefit of MCW-AF', fontweight='bold')

plt.suptitle('20% Label Noise: Harmful vs Beneficial Forgetting Analysis',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('experiments/fig7_noise_analysis.png', bbox_inches='tight', pad_inches=0.2)
plt.close()
print('\n[Saved] experiments/fig7_noise_analysis.png')

print("\nDone!")
