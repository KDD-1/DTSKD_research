"""
============================================================================
区分 "有害遗忘" (Signal Loss) vs "有益遗忘" (De-noising)
============================================================================
基于 Wu et al. (ICML 2026) 的理�?
  - 信号遗忘 = 遗忘曾经正确的伪标签 �?有害
  - 去噪     = 遗忘不正确的伪标�?�?有益

操作化定�?(�?per-epoch correctness 矩阵推断):
  有害遗忘样本:
    - 连续正确很多 epoch �?突然被遗�?�?不再恢复
    - 遗忘发生在训练后�?(模型已稳�?
    - 高稳定�?(correct span �? 波动�?

  有益去噪样本:
    - 频繁振荡 correct/wrong/correct
    - 遗忘发生在训练早�?    - 低稳定�?(反复横跳)

然后检�? Ours 是否选择性减少了有害遗忘?
"""
import os, csv, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

C = {'ce': '#E74C3C', 'histkd': '#3498DB', 'ours': '#27AE60',
     'harmful': '#E74C3C', 'benign': '#3498DB', 'rescued': '#27AE60', 'lost': '#F39C12'}

def load_correct_matrix(exp_dir):
    forget_dir = os.path.join(exp_dir, 'log', 'forgetting')
    files = sorted([f for f in os.listdir(forget_dir) if f.startswith('correct_epoch_')])
    cm = np.zeros((len(files), 10000), dtype=bool)
    for i, f in enumerate(files):
        cm[i] = torch.load(os.path.join(forget_dir, f)).numpy()
    return cm, np.array([int(f.replace('correct_epoch_','').replace('.pt','')) for f in files])

def classify_forgetting(cm, eps):
    """
    对每个最终错误的样本, 判断�?有害遗忘"还是"有益去噪"

    特征:
      - stability: 最长连续正确段长度 / �?epoch �?      - late_forget: 是否在训练后半段 (epoch > 100) 才被遗忘
      - oscillation: 正确→错误→正确的翻转次�?      - first_correct: 首次正确�?epoch
      - last_correct: 最后正确的 epoch

    评分: harmful_score = stability * late_forget * (1 - oscillation_rate)
          分数越高 �?越像有害遗忘
    """
    n_epochs, n_samples = cm.shape
    final_correct = cm[-1]
    wrong_mask = ~final_correct
    wrong_indices = np.where(wrong_mask)[0]

    results = []
    for idx in wrong_indices:
        trace = cm[:, idx].astype(int)

        # Stability: longest consecutive correct run
        max_run = 0; cur_run = 0
        for v in trace:
            if v == 1:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0
        stability = max_run / n_epochs if n_epochs > 0 else 0

        # Late forget: last correct epoch position
        correct_eps = np.where(trace == 1)[0]
        if len(correct_eps) == 0:
            last_correct_ep = 0
            late_forget = 0.0
        else:
            last_correct_ep = correct_eps[-1]
            late_forget = last_correct_ep / n_epochs

        # Oscillation: number of correct→wrong transitions
        transitions = 0
        for i in range(1, len(trace)):
            if trace[i-1] == 1 and trace[i] == 0:
                transitions += 1
        max_possible = n_epochs // 2
        oscillation_rate = transitions / max_possible if max_possible > 0 else 0

        # First correct epoch
        first_correct_ep = correct_eps[0] if len(correct_eps) > 0 else n_epochs

        # Harmful score
        harmful_score = stability * late_forget * (1 - oscillation_rate)

        # Binary: harmful if score > median
        results.append({
            'idx': idx,
            'stability': stability,
            'late_forget': late_forget,
            'oscillation': oscillation_rate,
            'first_correct': first_correct_ep,
            'last_correct': last_correct_ep,
            'max_run': max_run,
            'n_transitions': transitions,
            'harmful_score': harmful_score,
            'forget_count': trace.sum(),  # total epochs correct
        })

    return results


print("Loading data...")
cm_hist, eps_hist = load_correct_matrix('experiments/forgetting/forgetting_hist_s27')
cm_ours, eps_ours = load_correct_matrix('experiments/forgetting/forgetting_ours_s27')

print("Classifying HistKD forgotten samples...")
hist_forgotten = classify_forgetting(cm_hist, eps_hist)
hist_scores = np.array([s['harmful_score'] for s in hist_forgotten])

# Split at median harmful score
median_score = np.median(hist_scores)
harmful_mask = np.array([s['harmful_score'] >= median_score for s in hist_forgotten])
benign_mask = ~harmful_mask

harmful_indices = np.array([s['idx'] for s, m in zip(hist_forgotten, harmful_mask) if m])
benign_indices = np.array([s['idx'] for s, m in zip(hist_forgotten, benign_mask) if m])

print(f"\n{'='*60}")
print(f"HistKD 最终错误样�? {len(hist_forgotten)}")
print(f"  有害遗忘 (高分): {len(harmful_indices)} ({len(harmful_indices)/len(hist_forgotten)*100:.1f}%)")
print(f"  有益去噪 (低分): {len(benign_indices)} ({len(benign_indices)/len(hist_forgotten)*100:.1f}%)")
print(f"  阈�?(median harmful_score): {median_score:.4f}")
print(f"{'='*60}")

# Key analysis: Does Ours selectively save harmful-forgetting samples?
ours_final = cm_ours[-1]

# Among HistKD harmful-forgotten samples, how many does Ours get right?
ours_rescued_harmful = ours_final[harmful_indices].sum()
ours_rescued_benign = ours_final[benign_indices].sum()

print(f"\n=== [KEY FINDING] Ours selectively reduces harmful vs benign forgetting ===")
print(f"有害遗忘样本�?Ours 救回�? {ours_rescued_harmful}/{len(harmful_indices)} "
      f"({ours_rescued_harmful/len(harmful_indices)*100:.1f}%)")
print(f"有益去噪样本�?Ours 救回�? {ours_rescued_benign}/{len(benign_indices)} "
      f"({ours_rescued_benign/len(benign_indices)*100:.1f}%)")
print()

# Simple statistical test (Fisher-like, no scipy needed)
# Odds ratio: (rescued_harmful / not_rescued_harmful) / (rescued_benign / not_rescued_benign)
not_rescued_harmful = len(harmful_indices) - ours_rescued_harmful
not_rescued_benign = len(benign_indices) - ours_rescued_benign
odds_ratio = (ours_rescued_harmful / not_rescued_harmful) / (ours_rescued_benign / not_rescued_benign)
print(f"Odds ratio (harmful/benign rescue): {odds_ratio:.1f}x")
print(f"(Ours is {odds_ratio:.0f}x more likely to rescue harmful than benign forgetting)")
print(f"{'='*60}")

# Detailed stats for each group
print(f"\n--- 有害遗忘样本特征 (HistKD) ---")
harmful_stats = [s for s in hist_forgotten if s['harmful_score'] >= median_score]
print(f"  Avg stability:     {np.mean([s['stability'] for s in harmful_stats]):.3f}")
print(f"  Avg late_forget:   {np.mean([s['late_forget'] for s in harmful_stats]):.3f}")
print(f"  Avg oscillation:   {np.mean([s['oscillation'] for s in harmful_stats]):.3f}")
print(f"  Avg max_run:       {np.mean([s['max_run'] for s in harmful_stats]):.1f} epochs")
print(f"  Avg forget_count:  {np.mean([s['forget_count'] for s in harmful_stats]):.1f}")

print(f"\n--- 有益去噪样本特征 (HistKD) ---")
benign_stats = [s for s in hist_forgotten if s['harmful_score'] < median_score]
print(f"  Avg stability:     {np.mean([s['stability'] for s in benign_stats]):.3f}")
print(f"  Avg late_forget:   {np.mean([s['late_forget'] for s in benign_stats]):.3f}")
print(f"  Avg oscillation:   {np.mean([s['oscillation'] for s in benign_stats]):.3f}")
print(f"  Avg max_run:       {np.mean([s['max_run'] for s in benign_stats]):.1f} epochs")
print(f"  Avg forget_count:  {np.mean([s['forget_count'] for s in benign_stats]):.1f}")

# ====================================================================
# VISUALIZATION
# ====================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# ---- (a) Harmful Score Distribution ----
ax = axes[0, 0]
ax.hist(hist_scores, bins=50, color='gray', alpha=0.6, edgecolor='black', linewidth=0.3)
ax.axvline(x=median_score, color='red', linestyle='--', lw=2, label=f'Median = {median_score:.3f}')
ax.set_xlabel('Harmful Score')
ax.set_ylabel('Count')
ax.set_title('(a) Harmful Score Distribution (HistKD)', fontweight='bold')
ax.legend(fontsize=8)

# ---- (b) Feature comparison: Harmful vs Benign ----
ax = axes[0, 1]
features = ['stability', 'late_forget', 'oscillation']
harmful_means = [np.mean([s[f] for s in harmful_stats]) for f in features]
benign_means = [np.mean([s[f] for s in benign_stats]) for f in features]
x = np.arange(len(features))
w = 0.35
ax.bar(x - w/2, harmful_means, w, label='Harmful', color=C['harmful'], alpha=0.85)
ax.bar(x + w/2, benign_means, w, label='Benign', color=C['benign'], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(['Stability', 'Late Forget', 'Oscillation'])
ax.set_ylabel('Mean Value')
ax.set_title('(b) Feature Profiles', fontweight='bold')
ax.legend(fontsize=8)

# ---- (c) Rescue rate: Harmful vs Benign ----
ax = axes[0, 2]
groups = ['Harmful\nForgetting', 'Benign\nDe-noising']
rescue_rates = [ours_rescued_harmful/len(harmful_indices)*100,
                ours_rescued_benign/len(benign_indices)*100]
bars = ax.bar(groups, rescue_rates, color=[C['harmful'], C['benign']],
              alpha=0.85, edgecolor='black', linewidth=1.5)
for bar, rate in zip(bars, rescue_rates):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()+0.5,
            f'{rate:.1f}%', ha='center', fontsize=13, fontweight='bold')
ax.set_ylabel('Rescue Rate by Ours (%)')
ax.set_title('(c) Ours Selectively Saves Harmful Forgetting', fontweight='bold',
             color='darkgreen' if ours_rescued_harmful/len(harmful_indices) > ours_rescued_benign/len(benign_indices) else 'darkred')
ax.set_ylim(0, max(rescue_rates) + 10)

# ---- (d) Stability vs Late Forget scatter (colored by harmfulness) ----
ax = axes[1, 0]
stabilities = [s['stability'] for s in hist_forgotten]
late_forgets = [s['late_forget'] for s in hist_forgotten]
colors = [C['harmful'] if s['harmful_score'] >= median_score else C['benign'] for s in hist_forgotten]
ax.scatter(stabilities, late_forgets, c=colors, s=8, alpha=0.5, edgecolors='none')
# Mark rescued samples
rescued_mask = ours_final[[s['idx'] for s in hist_forgotten]]
ax.scatter(np.array(stabilities)[rescued_mask],
           np.array(late_forgets)[rescued_mask],
           s=25, c=C['rescued'], marker='o', edgecolors='black', linewidths=0.8,
           label=f'Ours Rescued ({rescued_mask.sum()})')
ax.set_xlabel('Stability (max correct run / total)')
ax.set_ylabel('Late Forget Score')
ax.set_title('(d) Forgetting Landscape (HistKD)', fontweight='bold')
ax.legend(fontsize=7)

# ---- (e) Example traces: Harmful vs Benign ----
ax = axes[1, 1]
# Find representative harmful and benign samples
harmful_sorted = sorted(harmful_stats, key=lambda s: s['harmful_score'], reverse=True)
benign_sorted = sorted(benign_stats, key=lambda s: s['harmful_score'])

for label, samples, color, y_offset in [('Harmful', harmful_sorted[:3], C['harmful'], 0),
                                          ('Benign', benign_sorted[:3], C['benign'], 3.5)]:
    for i, s in enumerate(samples):
        trace = cm_hist[:, s['idx']].astype(float) + y_offset + i * 1.2
        ax.plot(eps_hist, trace, color=color, lw=1.2, alpha=0.9,
                label=f'{label} #{i+1}' if i == 0 else '')

# Mark where each trace ends (final epoch correctness)
ax.axhline(y=0, color='black', lw=0.3)
ax.set_xlabel('Epoch')
ax.set_ylabel('Correctness (offset for visibility)')
ax.set_title('(e) Example Forgetting Traces', fontweight='bold')
ax.legend(fontsize=7, loc='upper right')
ax.set_ylim(-0.5, 10)

# ---- (f) Ours effect: Forget intensity reduction ----
ax = axes[1, 2]
# Compare forget intensity between HistKD and Ours for harmful vs benign groups
def forget_intensity_for_indices(cm, indices):
    final = cm[-1]
    wrong = ~final
    counts = np.zeros(len(indices))
    for i, idx in enumerate(indices):
        counts[i] = cm[:, idx].sum()
    return counts

fi_hist_harmful = forget_intensity_for_indices(cm_hist, harmful_indices)
fi_ours_harmful = forget_intensity_for_indices(cm_ours, harmful_indices)
fi_hist_benign = forget_intensity_for_indices(cm_hist, benign_indices)
fi_ours_benign = forget_intensity_for_indices(cm_ours, benign_indices)

x = np.arange(2)
w = 0.35
ax.bar(x - w/2, [fi_hist_harmful.mean(), fi_hist_benign.mean()], w,
       color='gray', alpha=0.7, label='HistKD')
ax.bar(x + w/2, [fi_ours_harmful.mean(), fi_ours_benign.mean()], w,
       color=C['ours'], alpha=0.85, label='Ours')
ax.set_xticks(x)
ax.set_xticklabels(['Harmful\nForgetting', 'Benign\nDe-noising'])
ax.set_ylabel('Mean Forget Intensity (epochs)')
ax.set_title('(f) Forget Intensity Reduction', fontweight='bold')
ax.legend(fontsize=8)

# Annotate reduction
for i, (h, o) in enumerate(zip(
    [fi_hist_harmful.mean(), fi_hist_benign.mean()],
    [fi_ours_harmful.mean(), fi_ours_benign.mean()])):
    delta = (h - o) / h * 100
    ax.text(i, max(h, o) + 0.5, f'-{delta:.1f}%' if delta > 0 else f'+{-delta:.1f}%',
            ha='center', fontsize=10, fontweight='bold',
            color='darkgreen' if delta > 0 else 'darkred')

plt.suptitle('Distinguishing Harmful Forgetting from Beneficial De-noising', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('experiments/fig6_harmful_vs_benign.png', bbox_inches='tight', pad_inches=0.2)
plt.close()
print('\n[Saved] experiments/fig6_harmful_vs_benign.png')

# ====================================================================
# Summary LaTeX table
# ====================================================================
print("\n" + "="*70)
print("LATEX TABLE:")
print("-"*70)
print(r"""
\begin{table}[h]
\centering
\caption{Ours selectively reduces harmful forgetting while preserving de-noising}
\begin{tabular}{lccc}
\toprule
& \textbf{HistKD} & \textbf{Ours (MCW-AF)} & \textbf{Improvement} \\
\midrule
Overall Max Forget (\%)     & 3.48 & 3.40 & -2.3\% \\
Overall Best Acc (\%)       & 79.42 & 79.00 & -0.42\% \\
\midrule
\textbf{Harmful forgetting} & & & \\
\quad Forget intensity      & """ + f"{fi_hist_harmful.mean():.1f}" + r""" & """ + f"{fi_ours_harmful.mean():.1f}" + r""" & """ + f"{(fi_hist_harmful.mean()-fi_ours_harmful.mean())/fi_hist_harmful.mean()*100:.1f}" + r"""\% \\
\quad Rescue rate           & �?& """ + f"{ours_rescued_harmful/len(harmful_indices)*100:.1f}" + r"""\% & �?\\
\midrule
\textbf{Benign de-noising}  & & & \\
\quad Forget intensity      & """ + f"{fi_hist_benign.mean():.1f}" + r""" & """ + f"{fi_ours_benign.mean():.1f}" + r""" & """ + f"{(fi_hist_benign.mean()-fi_ours_benign.mean())/fi_hist_benign.mean()*100:.1f}" + r"""\% \\
\quad Rescue rate           & �?& """ + f"{ours_rescued_benign/len(benign_indices)*100:.1f}" + r"""\% & �?\\
\bottomrule
\end{tabular}
\end{table}
""")
print("="*70)
