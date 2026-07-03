"""
============================================================================
50% 标签噪声实验 �?完整噪声梯度分析 (0% �?20% �?50%)
============================================================================
对比: clean / noise20 / noise50 �?HistKD vs MCW-AF v2
目标: 验证 "噪声�?�?�?AF 净收益 �? 趋势，确�?AF 首次超越 HistKD 的噪声阈�?"""

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

C = {
    'histkd': '#3498DB', 'ours': '#27AE60',
    'harmful': '#E74C3C', 'benign': '#3498DB',
    'rescued': '#27AE60', 'lost': '#F39C12',
    'clean': '#2ECC71', 'noise20': '#E67E22', 'noise50': '#E74C3C',
    'net_neg': '#E74C3C', 'net_pos': '#27AE60',
}

# ===================================================================
# 1. 数据加载
# ===================================================================
def load_correct_matrix(exp_dir):
    forget_dir = os.path.join(exp_dir, 'log', 'forgetting')
    files = sorted([f for f in os.listdir(forget_dir) if f.startswith('correct_epoch_')])
    n_epochs = len(files) // 2 if len(files) > 100 else len(files)
    if len(files) > 100:
        files = files[:200]
    cm = np.zeros((len(files), 10000), dtype=bool)
    for i, f in enumerate(files):
        cm[i] = torch.load(os.path.join(forget_dir, f)).numpy()
    return cm

def load_metrics(exp_dir):
    import csv
    epochs, vals, train_losses = [], [], []
    with open(os.path.join(exp_dir, 'log', 'metrics.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            vals.append(float(row['val_top1']))
            if 'train_loss' in row:
                train_losses.append(float(row['train_loss']))
    return epochs, vals, train_losses

def get_best_val(exp_dir):
    epochs, vals, _ = load_metrics(exp_dir)
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
        max_run = 0; cur_run = 0
        for v in trace:
            if v == 1:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0
        correct_eps = np.where(trace == 1)[0]
        last_correct_ep = correct_eps[-1] if len(correct_eps) > 0 else 0
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


def analyze_experiment_pair(hist_dir, ours_dir, label):
    """Analyze one (HistKD, Ours) pair: returns all key metrics"""
    cm_h = load_correct_matrix(hist_dir)
    cm_o = load_correct_matrix(ours_dir)

    hist_val, hist_ep = get_best_val(hist_dir)
    ours_val, ours_ep = get_best_val(ours_dir)

    # Forget classification
    hist_forgotten, hist_wrong_idx = classify_forgetting(cm_h)
    scores = np.array([s['harmful_score'] for s in hist_forgotten])
    median = np.median(scores)
    harmful_mask = scores >= median
    benign_mask = ~harmful_mask

    n_harmful = harmful_mask.sum()
    n_benign = benign_mask.sum()
    harmful_idx_arr = np.array([s['idx'] for s, m in zip(hist_forgotten, harmful_mask) if m])
    benign_idx_arr = np.array([s['idx'] for s, m in zip(hist_forgotten, benign_mask) if m])

    # Rescue analysis
    ours_final = cm_o[-1]
    rescued_harmful = ours_final[harmful_idx_arr].sum()
    rescued_benign = ours_final[benign_idx_arr].sum()

    # Lost: HistKD correct but Ours wrong
    hist_final = cm_h[-1]
    ours_wrong_idx_arr = np.where(~ours_final)[0]
    n_lost = hist_final[ours_wrong_idx_arr].sum()

    # Odds ratio
    not_h = n_harmful - rescued_harmful
    not_b = n_benign - rescued_benign
    odds = (rescued_harmful / not_h) / (rescued_benign / not_b) if not_b > 0 and not_h > 0 else float('inf')

    net = int(rescued_harmful + rescued_benign - n_lost)

    return {
        'label': label,
        'hist_val': hist_val, 'ours_val': ours_val,
        'hist_ep': hist_ep, 'ours_ep': ours_ep,
        'n_total_wrong': len(hist_forgotten),
        'n_harmful': int(n_harmful),
        'n_benign': int(n_benign),
        'harmful_pct': n_harmful / len(hist_forgotten) * 100,
        'rescued_harmful': int(rescued_harmful),
        'rescued_benign': int(rescued_benign),
        'harmful_rescue_rate': rescued_harmful / n_harmful * 100,
        'benign_rescue_rate': rescued_benign / n_benign * 100,
        'odds_ratio': odds,
        'n_lost': int(n_lost),
        'net_effect': net,
        'cm_h': cm_h, 'cm_o': cm_o,
        'hist_forgotten': hist_forgotten,
        'scores': scores, 'median': median,
    }


# ===================================================================
# 3. 主分�?# ===================================================================
print("=" * 75)
print("FULL NOISE GRADIENT ANALYSIS: 0% �?20% �?50% Label Noise")
print("=" * 75)

all_results = {}

# --- Clean ---
print("\n[1/3] Clean (0% noise)...")
try:
    all_results['0%'] = analyze_experiment_pair(
        'experiments/forgetting/forgetting_hist_s27',
        'experiments/forgetting/forgetting_ours_s27',
        'Clean')
    print(f"  HistKD: {all_results['0%']['hist_val']:.2f}%, "
          f"Ours: {all_results['0%']['ours_val']:.2f}%, "
          f"Δ: {all_results['0%']['ours_val']-all_results['0%']['hist_val']:+.2f}%")
except Exception as e:
    print(f"  SKIP: {e}")
    all_results['0%'] = None

# --- 20% Noise ---
print("\n[2/3] 20% Noise...")
try:
    all_results['20%'] = analyze_experiment_pair(
        'experiments/forgetting/noise/noise20_hist_s27',
        'experiments/forgetting/noise/noise20_ours_s27',
        '20% Noise')
    print(f"  HistKD: {all_results['20%']['hist_val']:.2f}%, "
          f"Ours: {all_results['20%']['ours_val']:.2f}%, "
          f"Δ: {all_results['20%']['ours_val']-all_results['20%']['hist_val']:+.2f}%")
except Exception as e:
    print(f"  SKIP: {e}")
    all_results['20%'] = None

# --- 50% Noise ---
print("\n[3/3] 50% Noise...")
try:
    all_results['50%'] = analyze_experiment_pair(
        'experiments/forgetting/noise/noise50_hist_s27',
        'experiments/forgetting/noise/noise50_ours_s27',
        '50% Noise')
    print(f"  HistKD: {all_results['50%']['hist_val']:.2f}%, "
          f"Ours: {all_results['50%']['ours_val']:.2f}%, "
          f"Δ: {all_results['50%']['ours_val']-all_results['50%']['hist_val']:+.2f}%")
except Exception as e:
    print(f"  SKIP: {e}")
    all_results['50%'] = None

# ===================================================================
# 4. 汇总对比表
# ===================================================================
print("\n" + "=" * 75)
print("SUMMARY: Noise Rate �?AF Benefit Gradient")
print("-" * 75)
header = f"{'Metric':<35}"
for key in ['0%', '20%', '50%']:
    header += f" {key:>12}"
print(header)
print("-" * 75)

rows = [
    ('HistKD Best Val', 'hist_val', '%', '{:.2f}'),
    ('Ours Best Val', 'ours_val', '%', '{:.2f}'),
    ('D (Ours - HistKD)', None, '%', None),  # special
    ('Harmful Forget Fraction', 'harmful_pct', '%', '{:.1f}'),
    ('Harmful Rescue Rate', 'harmful_rescue_rate', '%', '{:.1f}'),
    ('Benign Rescue Rate', 'benign_rescue_rate', '%', '{:.1f}'),
    ('Odds Ratio (H/B)', 'odds_ratio', 'x', '{:.1f}'),
    ('Lost Samples', 'n_lost', '', '{:d}'),
    ('Net Effect', 'net_effect', '', '{:+d}'),
]

for row_name, field, unit, fmt in rows:
    line = f"{row_name:<35}"
    for key in ['0%', '20%', '50%']:
        r = all_results.get(key)
        if r is None:
            line += f" {'�?:>12}"
        elif field is None:  # special: Δ
            delta = r['ours_val'] - r['hist_val']
            line += f" {delta:>+11.2f}%"
        elif unit == '%':
            val = r[field]
            line += f" {fmt.format(val):>11}{unit}"
        elif unit == 'x':
            val = r[field]
            if val == float('inf'):
                line += f" {'�?:>12}"
            else:
                line += f" {fmt.format(val):>11}x"
        else:
            val = r[field]
            line += f" {fmt.format(val):>12}"
    print(line)

print("-" * 75)

# Trend analysis
print("\n[TREND] Noise �?AF Benefit:")
valid_results = {k: v for k, v in all_results.items() if v is not None}
if len(valid_results) >= 2:
    noise_rates = []
    deltas = []
    net_effects = []
    for nr_str, r in valid_results.items():
        noise_rate = float(nr_str.replace('%', '')) / 100
        noise_rates.append(noise_rate)
        deltas.append(r['ours_val'] - r['hist_val'])
        net_effects.append(r['net_effect'])

    print(f"  Noise rates: {[f'{nr*100:.0f}%' for nr in noise_rates]}")
    print(f"  Δ Acc: {[f'{d:+.2f}%' for d in deltas]}")
    print(f"  Net Effect: {[f'{n:+d}' for n in net_effects]}")

# ===================================================================
# 5. 可视�?# ===================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

noise_keys = ['0%', '20%', '50%']
colors_noise = [C['clean'], C['noise20'], C['noise50']]

# ---- (a) Accuracy curves: all experiments ----
ax = axes[0, 0]
for key, color in zip(noise_keys, colors_noise):
    r = all_results.get(key)
    if r is None:
        continue
    for suffix, label, ls in [
        ('_hist_s27', 'HistKD', '--'),
        ('_ours_s27', 'Ours', '-'),
    ]:
        noise_dir = 'noise' + key.replace('%', '')
        if key == '0%':
            exp_dir = f'experiments/forgetting/forgetting{suffix}'
        else:
            exp_dir = f'experiments/forgetting/noise/{noise_dir}{suffix}'
        try:
            eps, vals, _ = load_metrics(exp_dir)
            ax.plot(eps, vals, color=color, linestyle=ls, lw=1.3,
                    alpha=0.85, label=f'{label} {key}')
        except:
            pass

for lr_ep in [100, 150]:
    ax.axvline(x=lr_ep, color='gray', linestyle=':', lw=0.6, alpha=0.4)
ax.set_xlabel('Epoch')
ax.set_ylabel('Val Top-1 (%)')
ax.set_title('(a) Accuracy Curves Across Noise Levels', fontweight='bold')
ax.legend(fontsize=6, loc='lower right', ncol=2)

# ---- (b) Δ Accuracy (Ours - HistKD) by noise level ----
ax = axes[0, 1]
for key, color in zip(noise_keys, colors_noise):
    r = all_results.get(key)
    if r is None:
        continue
    noise_dir = 'noise' + key.replace('%', '')
    if key == '0%':
        hist_dir = 'experiments/forgetting/forgetting_hist_s27'
        ours_dir = 'experiments/forgetting/forgetting_ours_s27'
    else:
        hist_dir = f'experiments/{noise_dir}_hist_s27'
        ours_dir = f'experiments/{noise_dir}_ours_s27'
    try:
        eps_h, vals_h, _ = load_metrics(hist_dir)
        eps_o, vals_o, _ = load_metrics(ours_dir)
        common = sorted(set(eps_h) & set(eps_o))
        delta = [vals_o[eps_o.index(e)] - vals_h[eps_h.index(e)] for e in common]
        ax.plot(common, delta, color=color, lw=1.8, label=f'{key} noise')
        ax.axhline(y=0, color='black', lw=0.5)
    except:
        pass

for lr_ep in [100, 150]:
    ax.axvline(x=lr_ep, color='gray', linestyle=':', lw=0.6, alpha=0.4)
ax.set_xlabel('Epoch')
ax.set_ylabel('Δ Val (Ours �?HistKD) [%]')
ax.set_title('(b) Ours vs HistKD Gap by Noise Level', fontweight='bold')
ax.legend(fontsize=8)

# ---- (c) Δ Gap at endpoint (scatter + trend line) ----
ax = axes[0, 2]
nr_vals = []
d_vals = []
for key, color in zip(noise_keys, colors_noise):
    r = all_results.get(key)
    if r is None:
        continue
    nr = float(key.replace('%', ''))
    delta = r['ours_val'] - r['hist_val']
    nr_vals.append(nr)
    d_vals.append(delta)
    marker = 'o'
    ax.scatter(nr, delta, c=color, s=200, zorder=5, edgecolors='black', linewidth=1.5, marker=marker)
    label_text = f'{delta:+.2f}%'
    ax.annotate(label_text, (nr, delta),
                textcoords="offset points", xytext=(0, 15 if delta < 0 else -15),
                ha='center', fontsize=10, fontweight='bold', color=color)

if len(nr_vals) >= 2:
    ax.plot(nr_vals, d_vals, 'o-', color='#555555', lw=2, markersize=0)
    # Trend line
    z = np.polyfit(nr_vals, d_vals, 1)
    x_fit = np.linspace(-5, 55, 100)
    y_fit = np.polyval(z, x_fit)
    ax.plot(x_fit, y_fit, '--', color='#888888', lw=1.5, alpha=0.7,
            label=f'Linear fit (crosses 0 at ~{(-z[1]/z[0]):.0f}% noise)' if z[0] > 0 else '')
    ax.legend(fontsize=8)

ax.axhline(y=0, color='black', lw=0.8)
ax.set_xlabel('Label Noise Rate (%)')
ax.set_ylabel('Δ Val (Ours �?HistKD) [%]')
ax.set_title('(c) Noise �?AF Benefit: Crossover Point', fontweight='bold')
ax.set_xlim(-5, 55)

# ---- (d) Harmful Rescue Rate vs Noise ----
ax = axes[1, 0]
x_pos = np.arange(len(noise_keys))
w = 0.3
for i, (key, color) in enumerate(zip(noise_keys, colors_noise)):
    r = all_results.get(key)
    if r is None:
        continue
    h_rate = r['harmful_rescue_rate']
    b_rate = r['benign_rescue_rate']
    ax.bar(i - w/2, h_rate, w, color=C['harmful'], alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.bar(i + w/2, b_rate, w, color=C['benign'], alpha=0.85, edgecolor='black', linewidth=1.2)
    ax.text(i - w/2, h_rate + 0.5, f'{h_rate:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.text(i + w/2, b_rate + 0.5, f'{b_rate:.1f}%', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(noise_keys)
ax.set_ylabel('Rescue Rate (%)')
ax.set_title('(d) Selective Rescue: Harmful (Red) vs Benign (Blue)', fontweight='bold')
# Legend
from matplotlib.patches import Patch
ax.legend([Patch(facecolor=C['harmful'], alpha=0.85), Patch(facecolor=C['benign'], alpha=0.85)],
          ['Harmful Forget', 'Benign Denoising'], fontsize=9)
ax.set_ylim(0, max(
    all_results.get('0%', {}).get('harmful_rescue_rate', 0) if all_results.get('0%') else 0,
    all_results.get('20%', {}).get('harmful_rescue_rate', 0) if all_results.get('20%') else 0,
    all_results.get('50%', {}).get('harmful_rescue_rate', 0) if all_results.get('50%') else 0,
    30
) + 10)

# ---- (e) Net Effect vs Noise ----
ax = axes[1, 1]
net_vals = []
for key, color in zip(noise_keys, colors_noise):
    r = all_results.get(key)
    if r is None:
        continue
    net = r['net_effect']
    net_vals.append(net)
    bar_color = C['net_pos'] if net >= 0 else C['net_neg']
    bar = ax.bar(key, net, color=bar_color, alpha=0.85, edgecolor='black', linewidth=1.5, width=0.5)
    ax.text(bar[0].get_x() + bar[0].get_width()/2., net,
            f'{net:+d}', ha='center', fontsize=13, fontweight='bold',
            va='bottom' if net >= 0 else 'top')

ax.axhline(y=0, color='black', lw=0.8)
ax.set_ylabel('Net Effect (Rescued �?Lost)')
ax.set_title('(e) Net Benefit of MCW-AF', fontweight='bold')

# ---- (f) Harmful Score Distribution comparison ----
ax = axes[1, 2]
for key, color in zip(noise_keys, colors_noise):
    r = all_results.get(key)
    if r is None:
        continue
    ax.hist(r['scores'], bins=35, color=color, alpha=0.3, edgecolor=color,
            linewidth=0.5, label=f'{key} (n={len(r["scores"])})', density=True)
    ax.axvline(x=r['median'], color=color, linestyle='--', lw=1.5, alpha=0.7)

ax.set_xlabel('Harmful Score')
ax.set_ylabel('Density')
ax.set_title('(f) Harmful Score Distribution by Noise', fontweight='bold')
ax.legend(fontsize=7)

plt.suptitle('50% Label Noise: Complete Noise Gradient Analysis (0% �?20% �?50%)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('experiments/fig8_noise_gradient.png', bbox_inches='tight', pad_inches=0.2)
plt.close()
print('\n[Saved] experiments/fig8_noise_gradient.png')

# ===================================================================
# 6. Key Conclusions
# ===================================================================
print("\n" + "=" * 75)
print("KEY CONCLUSIONS")
print("=" * 75)

# Check if 50% is complete
if all_results.get('50%') is not None:
    delta_50 = all_results['50%']['ours_val'] - all_results['50%']['hist_val']
    net_50 = all_results['50%']['net_effect']

    print(f"\n  50% Noise:")
    print(f"    Δ Acc: {delta_50:+.2f}%")
    print(f"    Net Effect: {net_50:+d}")

    if delta_50 > 0:
        print(f"\n  �?MCW-AF首次超越HistKD! Δ = {delta_50:+.2f}%")
        print(f"  �?50% noise is the crossover point where AF benefit > AF cost")
    elif delta_50 < 0:
        # Extrapolate crossover
        deltas_list = []
        nrs_list = []
        for key in ['0%', '20%', '50%']:
            r = all_results.get(key)
            if r is not None:
                deltas_list.append(r['ours_val'] - r['hist_val'])
                nrs_list.append(float(key.replace('%', '')))
        if len(deltas_list) >= 2:
            z = np.polyfit(nrs_list, deltas_list, 1)
            if abs(z[0]) > 1e-6:
                crossover = -z[1] / z[0]
                print(f"\n  Δ still negative at 50%, but improving.")
                print(f"  Linear extrapolation: crossover at ~{crossover:.0f}% noise")

print("\nDone!")
