"""
============================================================================
Noise Gradient Visualization: Noise Rate -> MCW-AF Benefit
============================================================================
Clean figure showing the crossover from AF-loses to AF-wins as noise increases.
"""

import numpy as np
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

# ===================================================================
# Data (from completed experiments)
# ===================================================================
noise_rates = [0, 20, 50]
noise_labels = ['0% (Clean)', '20%', '50%']

histkd_val  = [79.42, 75.57, 65.68]
ours_val    = [79.00, 75.46, 66.01]
deltas      = [-0.42, -0.11, +0.33]
harmful_rescue = [27.6, 32.0, 32.9]
benign_rescue  = [2.7, 5.4, 4.2]
odds_ratios    = [13.6, 8.2, 11.2]
lost_samples   = [363, 471, 592]
net_effects    = [-41, -12, +47]

colors_noise = ['#2ECC71', '#E67E22', '#E74C3C']

# ===================================================================
# Figure: 4 panels
# ===================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# ---- (a) Accuracy bars: HistKD vs Ours ----
ax = axes[0, 0]
x = np.arange(len(noise_labels))
w = 0.3
bars_h = ax.bar(x - w/2, histkd_val, w, color='#3498DB', alpha=0.85,
                edgecolor='black', linewidth=1.2, label='HistKD')
bars_o = ax.bar(x + w/2, ours_val, w, color='#E74C3C', alpha=0.85,
                edgecolor='black', linewidth=1.2, label='Ours (MCW-AF)')

for bar, val in zip(bars_h, histkd_val):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{val:.2f}', ha='center', fontsize=10, fontweight='bold', color='#2471A3')
for bar, val in zip(bars_o, ours_val):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{val:.2f}', ha='center', fontsize=10, fontweight='bold', color='#C0392B')

# Add delta labels
for i, d in enumerate(deltas):
    color = '#27AE60' if d > 0 else '#E74C3C'
    sign = '+' if d > 0 else ''
    y_pos = max(histkd_val[i], ours_val[i]) + 2.5
    ax.annotate(f'Delta = {sign}{d:.2f}%', (x[i], y_pos), ha='center',
                fontsize=11, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.9))

ax.set_xticks(x)
ax.set_xticklabels(noise_labels, fontsize=11)
ax.set_ylabel('Val Top-1 Accuracy (%)', fontsize=11)
ax.set_title('(a) Accuracy: HistKD vs Ours by Noise Level', fontweight='bold', fontsize=12)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(50, 87)

# ---- (b) Delta accuracy gradient + crossover ----
ax = axes[0, 1]
noise_arr = np.array(noise_rates)
delta_arr = np.array(deltas)

# Zero line
ax.axhline(y=0, color='black', lw=1.2, linestyle='-', alpha=0.4)

# Fill region: red below zero, green above
ax.fill_between([-5, 55], -1, 0, alpha=0.06, color='#E74C3C')
ax.fill_between([-5, 55], 0, 1, alpha=0.06, color='#27AE60')

# Bars
bar_colors = ['#E74C3C' if d < 0 else '#27AE60' for d in deltas]
bars = ax.bar(noise_rates, deltas, width=8, color=bar_colors, alpha=0.85,
              edgecolor='black', linewidth=1.5, zorder=3)

for bar, d in zip(bars, deltas):
    sign = '+' if d > 0 else ''
    y_offset = 0.06 if d >= 0 else -0.06
    va = 'bottom' if d >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2., d + y_offset,
            f'{sign}{d:.2f}%', ha='center', fontsize=13, fontweight='bold',
            va=va, color='#27AE60' if d > 0 else '#E74C3C')

# Linear fit + crossover
z = np.polyfit(noise_rates, deltas, 1)
x_fit = np.linspace(-8, 58, 100)
y_fit = np.polyval(z, x_fit)
crossover = -z[1] / z[0]
ax.plot(x_fit, y_fit, '--', color='#7F8C8D', lw=2, alpha=0.8)
ax.scatter([crossover], [0], marker='D', s=200, c='#8E44AD', zorder=5,
           edgecolors='black', linewidth=1.5)
ax.annotate(f'Crossover\n~{crossover:.0f}% noise',
            (crossover, 0), textcoords="offset points",
            xytext=(15, 25), ha='center', fontsize=11, fontweight='bold',
            color='#8E44AD',
            arrowprops=dict(arrowstyle='->', color='#8E44AD', lw=1.5))

ax.set_xlabel('Label Noise Rate (%)', fontsize=11)
ax.set_ylabel('Delta Accuracy (Ours - HistKD) [%]', fontsize=11)
ax.set_title('(b) Noise Rate -> AF Benefit (Crossover)', fontweight='bold', fontsize=12)
ax.set_xlim(-8, 58)
ax.set_ylim(-0.9, 0.8)

# Annotation labels
ax.text(-4, 0.63, 'AF Wins', fontsize=11, fontweight='bold', color='#27AE60', alpha=0.7)
ax.text(-4, -0.7, 'AF Loses', fontsize=11, fontweight='bold', color='#E74C3C', alpha=0.7)

# ---- (c) Net Effect bars ----
ax = axes[1, 0]
net_colors = ['#27AE60' if n >= 0 else '#E74C3C' for n in net_effects]
bars_net = ax.bar(noise_labels, net_effects, width=0.45, color=net_colors, alpha=0.85,
                  edgecolor='black', linewidth=1.5)
ax.axhline(y=0, color='black', lw=1, alpha=0.5)

for bar, val in zip(bars_net, net_effects):
    sign = '+' if val > 0 else ''
    va = 'bottom' if val >= 0 else 'top'
    y_off = 1.5 if val >= 0 else -1.5
    ax.text(bar.get_x() + bar.get_width()/2., val + y_off,
            f'{sign}{val}', ha='center', fontsize=16, fontweight='bold',
            va=va, color='#27AE60' if val > 0 else '#E74C3C')

ax.set_ylabel('Net Effect (Rescued - Lost)', fontsize=11)
ax.set_title('(c) Net Benefit of MCW-AF', fontweight='bold', fontsize=12)

# ---- (d) Harmful Rescue + Odds Ratio dual-axis ----
ax = axes[1, 1]
x_pos = np.arange(len(noise_labels))
w2 = 0.3

bars_hr = ax.bar(x_pos - w2/2, harmful_rescue, w2, color='#E74C3C', alpha=0.85,
                 edgecolor='black', linewidth=1.2, label='Harmful Rescue Rate (%)')
bars_br = ax.bar(x_pos + w2/2, benign_rescue, w2, color='#3498DB', alpha=0.85,
                 edgecolor='black', linewidth=1.2, label='Benign Rescue Rate (%)')
for bar, val in zip(bars_hr, harmful_rescue):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.4,
            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#C0392B')
for bar, val in zip(bars_br, benign_rescue):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.4,
            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#2471A3')

ax2 = ax.twinx()
ax2.plot(x_pos, odds_ratios, 's-', color='#8E44AD', lw=2.5, markersize=12,
         markerfacecolor='#8E44AD', markeredgecolor='black', markeredgewidth=1,
         label='Odds Ratio (H/B)', zorder=5)
for i, o in enumerate(odds_ratios):
    ax2.text(x_pos[i], o + 0.3, f'{o:.1f}x', ha='center', fontsize=10,
             fontweight='bold', color='#6C3483')

ax.set_xticks(x_pos)
ax.set_xticklabels(noise_labels, fontsize=11)
ax.set_ylabel('Rescue Rate (%)', fontsize=11)
ax2.set_ylabel('Odds Ratio (Harmful / Benign)', fontsize=11, color='#8E44AD')
ax.set_title('(d) Selective Rescue: Harmful vs Benign', fontweight='bold', fontsize=12)
ax.set_ylim(0, 42)

# Combined legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E74C3C', alpha=0.85, label='Harmful Rescue'),
    Patch(facecolor='#3498DB', alpha=0.85, label='Benign Rescue'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#8E44AD',
               markersize=10, markeredgecolor='black', markeredgewidth=1,
               label='Odds Ratio (H/B)')
]
ax.legend(handles=legend_elements, fontsize=9, loc='upper left')

# ===================================================================
# Final
# ===================================================================
plt.suptitle('MCW-AF Noise Gradient: Label Noise Rate -> Anti-Forgetting Benefit',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig('experiments/fig_noise_gradient.png', bbox_inches='tight', pad_inches=0.2)
plt.close()
print('[Saved] experiments/fig_noise_gradient.png')
print()
print('=' * 60)
print('Key numbers:')
print(f'  Clean (0%):  Delta = -0.42%, Net = -41  (AF loses)')
print(f'  Noise (20%): Delta = -0.11%, Net = -12  (AF close to tie)')
print(f'  Noise (50%): Delta = +0.33%, Net = +47  (AF wins!)')
print(f'  Crossover ~ {crossover:.0f}% noise')
print('=' * 60)
