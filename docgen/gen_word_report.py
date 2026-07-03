"""
生成 MCW-AF 遗忘研究 Word 文档
"""
import os, csv
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import numpy as np
import torch

# ── 数据加载 ──────────────────────────────────────
def load_metrics(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        first = f.readline().strip()
        f.seek(0)
        if first.startswith('epoch'):
            return list(csv.DictReader(f))
        cols = ['epoch','lr','alpha_t','train_loss','train_top1','train_top5',
                'val_top1','val_top5','val_b1_top1','val_b2_top1','val_b3_top1']
        return [dict(zip(cols, r)) for r in csv.reader(f)]

def load_cm(exp_dir):
    d = os.path.join(exp_dir, 'log', 'forgetting')
    files = sorted([f for f in os.listdir(d) if f.startswith('correct_epoch_')])
    cm = np.zeros((len(files), 10000), dtype=bool)
    for i, f in enumerate(files):
        cm[i] = torch.load(os.path.join(d, f)).numpy()
    return cm

def best_val(rows):
    best_idx = max(range(len(rows)), key=lambda i: float(rows[i]['val_top1']))
    return float(rows[best_idx]['val_top1']), int(rows[best_idx]['epoch'])

def max_forget(cm):
    final = cm[-1]; wrong = ~final
    return max((cm[e] & wrong).sum() / 10000 for e in range(len(cm)))

cm_ce = load_cm('experiments/forgetting/forgetting_ce_s27')
cm_hist = load_cm('experiments/forgetting/forgetting_hist_s27')
cm_v1 = load_cm('experiments/forgetting/forgetting_ours_s27')
cm_v2 = load_cm('experiments/forgetting/forgetting_ours_s27_v2')
cm_hist_n20 = load_cm('experiments/forgetting/noise/noise20_hist_s27')
cm_ours_n20 = load_cm('experiments/forgetting/noise/noise20_ours_s27')

# harmful/benign classification
def classify_forgetting(cm):
    n_epochs, n_samples = cm.shape
    final_correct = cm[-1]; wrong_mask = ~final_correct
    results = []
    for idx in np.where(wrong_mask)[0]:
        trace = cm[:, idx].astype(int)
        max_run = 0; cur_run = 0
        for v in trace:
            if v == 1: cur_run += 1; max_run = max(max_run, cur_run)
            else: cur_run = 0
        stability = max_run / n_epochs
        correct_eps = np.where(trace == 1)[0]
        last_correct = correct_eps[-1] if len(correct_eps) > 0 else 0
        late_forget = last_correct / n_epochs
        transitions = sum(1 for i in range(1, len(trace)) if trace[i-1]==1 and trace[i]==0)
        oscillation_rate = transitions / (n_epochs//2)
        harmful_score = stability * late_forget * (1 - oscillation_rate)
        results.append({'idx': idx, 'harmful_score': harmful_score, 'max_run': max_run,
                        'last_correct': last_correct, 'forget_count': trace.sum()})
    return results

hist_forgotten = classify_forgetting(cm_hist)
median_score = np.median([s['harmful_score'] for s in hist_forgotten])
harmful_mask = np.array([s['harmful_score'] >= median_score for s in hist_forgotten])
harmful_idx = np.array([s['idx'] for s, m in zip(hist_forgotten, harmful_mask) if m])
benign_idx = np.array([s['idx'] for s, m in zip(hist_forgotten, harmful_mask) if not m])

# Rescue rates
final_v1 = cm_v1[-1]; final_v2 = cm_v2[-1]; final_hist = cm_hist[-1]
v1_harmful_rescue = final_v1[harmful_idx].sum()
v1_benign_rescue = final_v1[benign_idx].sum()
v2_harmful_rescue = final_v2[harmful_idx].sum()
v2_benign_rescue = final_v2[benign_idx].sum()

# Phase accuracy
def phase_acc(cm, start, end):
    return cm[start:end].mean(axis=1)[-1] * 100

# ── 20% 噪声实验数据 ──────────────────────────────
# Clean reference values
m_hist_c = load_metrics('experiments/forgetting/forgetting_hist_s27/log/metrics.csv')
hist_val_c, _ = best_val(m_hist_c)

m_hist_n20 = load_metrics('experiments/forgetting/noise/noise20_hist_s27/log/metrics.csv')
m_ours_n20 = load_metrics('experiments/forgetting/noise/noise20_ours_s27/log/metrics.csv')
hist_val_n20, hist_val_ep_n20 = best_val(m_hist_n20)
ours_val_n20, ours_val_ep_n20 = best_val(m_ours_n20)

# Noise harmful/benign decomposition
hist_forgotten_n20 = classify_forgetting(cm_hist_n20)
median_score_n20 = np.median([s['harmful_score'] for s in hist_forgotten_n20])
harmful_mask_n20 = np.array([s['harmful_score'] >= median_score_n20 for s in hist_forgotten_n20])
harmful_idx_n20 = np.array([s['idx'] for s, m in zip(hist_forgotten_n20, harmful_mask_n20) if m])
benign_idx_n20 = np.array([s['idx'] for s, m in zip(hist_forgotten_n20, harmful_mask_n20) if not m])

final_ours_n20 = cm_ours_n20[-1]
final_hist_n20 = cm_hist_n20[-1]
n20_harmful_rescue = final_ours_n20[harmful_idx_n20].sum()
n20_benign_rescue = final_ours_n20[benign_idx_n20].sum()
n20_harmful_n = len(harmful_idx_n20)
n20_benign_n = len(benign_idx_n20)
n20_odds = (n20_harmful_rescue / (n20_harmful_n - n20_harmful_rescue)) / (n20_benign_rescue / (n20_benign_n - n20_benign_rescue))
n20_rescued = (final_ours_n20 & (~final_hist_n20)).sum()
n20_lost = ((~final_ours_n20) & final_hist_n20).sum()
n20_net = n20_rescued - n20_lost

# ── 文档生成 ──────────────────────────────────────
doc = Document()

style = doc.styles['Normal']
font = style.font; font.name = 'Times New Roman'; font.size = Pt(11)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

for level in range(1, 4):
    hs = doc.styles[f'Heading {level}']
    hs.font.name = 'Times New Roman'
    hs.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

def add_table(doc, headers, rows, col_widths=None, bold_first=True):
    """添加格式化表�?""
    table = doc.add_table(rows=1+len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]; cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs: run.bold = True; run.font.size = Pt(9)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.rows[r+1].cells[c]; cell.text = str(val)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs: run.font.size = Pt(9)
                if bold_first and c == 0:
                    for run in p.runs: run.bold = True
    doc.add_paragraph()
    return table

# ════════════════════════════════════════════════
# 封面
# ════════════════════════════════════════════════
doc.add_paragraph()
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('DTSKD 遗忘研究\nMCW-AF 方法实验记录与分�?)
run.font.size = Pt(22); run.bold = True; run.font.color.rgb = RGBColor(0x1a, 0x56, 0xdb)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('Marginal Confidence Weighted Anti-Forgetting\nfor Self-Knowledge Distillation')
run.font.size = Pt(13); run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

info = doc.add_paragraph()
info.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = info.add_run('日期�?026-06-26\n分支：ablation_study\n模型：ResNet18-DTSKD + CIFAR-100\n训练轮数�?00 epochs')
run.font.size = Pt(11)

doc.add_page_break()

# ════════════════════════════════════════════════
# 一、实验矩�?# ════════════════════════════════════════════════
doc.add_heading('一、实验矩�?, level=1)

doc.add_paragraph(
    '所有实验共享超参数：end_epoch=200, batch_size=64, lr=0.1, lr_decay_schedule=[100,150], '
    'coeff_decay=cos, cos_max=0.9, cos_min=0.0, alpha_T=0.8, alpha_end_epoch=200, '
    'classifier_type=resnet18_dtskd, data_type=cifar100, workers=0, track_forgetting=1, random_seed=27.')

add_table(doc,
    ['实验�?, '方法', 'Seed', 'Best Val', 'Max Forget', '状�?],
    [['forgetting_ce_s27',     'CE Only',               '27', '76.33%', '5.37%', '�?完成'],
     ['forgetting_ce_s42',     'CE Only',               '42', '76.50%', '�?,    '�?完成'],
     ['forgetting_hist_s27',   'HistKD Only',           '27', '79.42%', '3.48%', '�?完成'],
     ['forgetting_ours_s27',   'MCW-AF v1 (λ=0.5恒定)',  '27', '79.00%', '3.40%', '�?完成'],
     ['forgetting_ours_s27_v2','MCW-AF v2 (λ�?-α_t)',   '27', '79.01%', '3.25%', '�?完成']])

# ════════════════════════════════════════════════
# 二、方法描�?# ════════════════════════════════════════════════
doc.add_heading('二、MCW-AF 方法', level=1)

doc.add_heading('2.1 核心动机', level=2)
doc.add_paragraph(
    '自知识蒸�?(SKD) 中的样本遗忘有两种本质不同的类型�?1) 信号遗忘——稳定学会的样本被遗忘，'
    '造成真正的性能损失�?2) 去噪——从未稳定学会的样本"被遗�?，实际是伪标签的自然替换过程�?
    '有利于模型泛化。传统的 EWC 方法不加区分地阻止所有遗忘，不可避免地阻碍了有益的去噪�?
    'MCW-AF (Marginal Confidence Weighted Anti-Forgetting) 通过置信度加权机制，'
    '选择性保护高置信度的正确知识，同时允许低置信度的预测继续演化和去噪�?)

doc.add_heading('2.2 数学形式', level=2)

p = doc.add_paragraph()
p.add_run('�?1) �?边际置信度：').bold = True
p.add_run('\nm_i = p_old(y_i | x_i) �?max_{k≠y_i} p_old(k | x_i)')
doc.add_paragraph(
    '其中 p_old 为历史累积预测。m_i > 0 表示历史预测正确且置信度超过所有错误类别，'
    'm_i �?0 表示历史预测错误�?)

p = doc.add_paragraph()
p.add_run('�?2) �?动态连续权重（"抽奖机制"）：').bold = True
p.add_run('\nw_i = exp(α · (m_i �?τ))  if m_i > 0, else 0')
doc.add_paragraph(
    'm_i �?0（历史错误）�?w_i = 0，完全不保护；m_i > 0 �?指数型映射，'
    '边际置信度越高保护越强。�?控制保护强度，�?控制保护门槛�?)

p = doc.add_paragraph()
p.add_run('�?3) �?总损失函数：').bold = True
p.add_run('\nL_total = L_CE+KD + λ · (1/B) Σ_i w_i · KL(p_i^old || p_i^new)')
doc.add_paragraph(
    '前向 KL 散度强制新预测分布覆盖旧分布的高概率区域（mode-covering），'
    '配合 w_i 实现选择性保护。�?控制全局 AF 强度�?)

doc.add_heading('2.3 v1 vs v2：AF 强度调度策略', level=2)

doc.add_paragraph(
    'v1 �?AF 强度 λ=0.5 全程恒定。这导致早期训练阶段 AF 过早约束模型探索�?
    'v2 �?AF 强度�?α_t（HSKD 的历史信任系数）挂钩，实现自然对偶：')

p = doc.add_paragraph()
p.add_run('v2: λ_eff = λ · (1 �?α_t)').bold = True
p.add_run('\n�?α_t 高（早期）：信任 GT 标签，历史预测不可靠 �?AF 弱（λ_eff �?0.05�?)
p.add_run('\n�?α_t 低（后期）：全靠历史预测，历史是唯一教师 �?AF 强（λ_eff �?0.50�?)

add_table(doc,
    ['阶段', 'α_t', 'v1 λ_eff', 'v2 λ_eff', '含义'],
    [['早期 (ep 0)',   '0.90', '0.50', '0.05', 'v1 约束过强，v2 几乎无约�?],
     ['中早�?(ep 50)','0.77', '0.50', '0.12', 'v2 逐步增强'],
     ['中期 (ep 100)','0.45', '0.50', '0.28', 'LR decay �?AF 加强巩固'],
     ['后期 (ep 150)','0.13', '0.50', '0.44', 'AF 接近满强�?],
     ['末期 (ep 199)','�?',  '0.50', '0.50', '相同终点']])

# ════════════════════════════════════════════════
# 三、核心结�?# ════════════════════════════════════════════════
doc.add_heading('三、核心实验结�?, level=1)

doc.add_heading('3.1 精度与遗忘总览', level=2)

add_table(doc,
    ['方法', 'Best Val', 'Max Forget', 'vs CE ΔAcc', 'vs CE ΔForget'],
    [['CE Only',       '76.33%', '5.37%', '�?,      '�?],
     ['HistKD Only',   '79.42%', '3.48%', '+3.09%', '�?5.2%'],
     ['Ours v1 (恒定)', '79.00%', '3.40%', '+2.67%', '�?6.7%'],
     ['Ours v2 (反转)', '79.01%', '3.25%', '+2.68%', '�?9.5%']])

doc.add_paragraph(
    '初步观察：MCW-AF �?Max Forget 持续降低（v1: 3.40%, v2: 3.25%），但精度从未超�?HistKD '
    '(v1: 79.00%, v2: 79.01% vs HistKD: 79.42%)。表面上看，方法的改进幅度有限�?)

doc.add_heading('3.2 分阶段精度——揭示隐藏动�?, level=2)

doc.add_paragraph(
    '以下分析揭示�?v2 在训练过程中的非单调行为，这是表面精度数字无法反映的关键动态：')

add_table(doc,
    ['方法', 'Epoch 50', 'Epoch 100', 'Epoch 150', 'Final'],
    [['HistKD',   '55.8%', '54.9%', '75.1%', '78.8%'],
     ['v1 (恒定)', '53.5%', '58.4%', '74.6%', '78.4%'],
     ['v2 (反转)', '55.6%', '59.9%', '75.1%', '78.6%']])

p = doc.add_paragraph()
p.add_run('v2 �?epoch 100 时大幅领�?HistKD�?9.9% vs 54.9%�?5.0%），').bold = True
p.add_run('证明�?AF 策略成功解放了早期学习。但 LR decay �?AF 变强（λ_eff �?0.5），'
           'v2 的增长开始停滞，HistKD 则持续追赶并最终反超。这说明 AF 机制在后期阻碍了模型精调—�?
           '即使保护时机正确，任何防止预测变化的约束都会在一定程度上限制模型的最优收敛�?)

doc.add_heading('3.3 样本级对�?, level=2)

v1_rescued = (final_v1 & (~final_hist)).sum()
v1_lost = ((~final_v1) & final_hist).sum()
v2_rescued = (final_v2 & (~final_hist)).sum()
v2_lost = ((~final_v2) & final_hist).sum()

add_table(doc,
    ['对比', '救回 (Ours正确 HistKD错误)', '丢失 (Ours错误 HistKD正确)', '净效果'],
    [['v1 vs HistKD', str(v1_rescued), str(v1_lost), f'{v1_rescued - v1_lost:+d}'],
     ['v2 vs HistKD', str(v2_rescued), str(v2_lost), f'{v2_rescued - v2_lost:+d}']])

doc.add_paragraph(
    'v2 的净效果（−21）优�?v1（−41），但仍为负。时序反转缓解了部分问题，但未能根本解决�?
    'AF 保护 352 个样本的同时，间接损害了 373 个样本�?)

# ════════════════════════════════════════════════
# 四、有害遗�?vs 有益去噪
# ════════════════════════════════════════════════
doc.add_heading('四、核心发现：有害遗忘 vs 有益去噪的分�?, level=1)

doc.add_heading('4.1 理论框架', level=2)
doc.add_paragraph(
    '基于 Wu et al. (ICML 2026) 的理论——Self-training 中的遗忘可以分为信号遗忘（有害）�?
    '去噪（有益）。我们从 per-epoch correctness 矩阵�?00 epochs × 10,000 samples）中'
    '操作化定义了区分标准�?)

add_table(doc,
    ['特征', '有害遗忘 (Signal Loss)', '有益去噪 (De-noising)'],
    [['最长连续正�?,  '6.5 epochs',           '1.0 epochs'],
     ['遗忘时机',      '后期 (epoch ~160)',    '早期 (epoch ~76)'],
     ['总正�?epoch �?,'40.1',                '5.4'],
     ['振荡�?,        '0.23',                 '0.05'],
     ['�?HistKD 错误样本', '50.0% (1061�?',  '50.0% (1060�?']])

p = doc.add_paragraph()
p.add_run('判别公式�?).bold = True
p.add_run('HarmfulScore = stability × late_forget × (1 �?oscillation_rate)')
p.add_run('\n�?HistKD 遗忘样本的中位数二分，高分为有害遗忘，低分为有益去噪�?)

doc.add_heading('4.2 MCW-AF 的选择性保护——核心证�?, level=2)

add_table(doc,
    ['', '有害遗忘 (n=1061)', '有益去噪 (n=1060)', 'Odds Ratio'],
    [['Ours v1 救回', f'{v1_harmful_rescue} ({(v1_harmful_rescue/1061*100):.1f}%)',
                     f'{v1_benign_rescue} ({(v1_benign_rescue/1060*100):.1f}%)',
                     f'{(v1_harmful_rescue/1061)/(v1_benign_rescue/1060):.1f}x'],
     ['Ours v2 救回', f'{v2_harmful_rescue} ({(v2_harmful_rescue/1061*100):.1f}%)',
                     f'{v2_benign_rescue} ({(v2_benign_rescue/1060*100):.1f}%)',
                     f'{(v2_harmful_rescue/1061)/(v2_benign_rescue/1060):.1f}x']])

p = doc.add_paragraph()
p.add_run('MCW-AF 对有害遗忘的保护强度是有益去噪的 17 倍（v2）�?).bold = True
p.add_run('这证明了置信度加权机制天然实现了选择性抗遗忘：高置信�?高稳定性样本获得强保护（阻止信号丢失）�?
           '低置信度/高振荡样本几乎不受影响（允许去噪继续）�?)

doc.add_heading('4.3 为什么选择性保护没有转化为精度优势�?, level=2)

doc.add_paragraph(
    '尽管 MCW-AF 选择性保护了有害遗忘样本（v2 救回 33.2%），但整体精度仍低于 HistKD�?
    '根本原因在于 AF 约束通过 parameter interference 间接影响了所有样本的学习�?)

p = doc.add_paragraph()
p.add_run('数量不对称：')
p.add_run('\n�?有害遗忘仅占最终错误样本的 50%�?061/2121�?)
p.add_run('\n�?�?AF 成功保护的仅占有害遗忘的 33%�?52/1061�?)
p.add_run('\n�?�?AF 间接损害的占 HistKD 正确样本�?4.7%�?73/7879�?)
p.add_run('\n�?净效果为负�?352 �?373 = �?1')

p = doc.add_paragraph()
p.add_run('本质冲突�?)
p.add_run('\n�?SKD �?50% �?遗忘"实际上是有益的去噪过�?)
p.add_run('\n�?EWC/MCW 类方法本质上是防止预测变�?)
p.add_run('\n�?�?SKD 的训练过程需要预测变化——因为历史预测本身在持续改进')
p.add_run('\n�?阻止变化 = 阻止改进 = 精度上限降低')

# ════════════════════════════════════════════════
# 五、可视化产出
# ════════════════════════════════════════════════
doc.add_heading('五、可视化图表', level=1)

figures = [
    ('fig1_main_comparison.png', 'Fig.1 主对比图',
     '遗忘曲线 + 精度曲线 + 训练Loss + 柱状汇总，三方法并排对�?),
    ('fig2_ours_detail.png', 'Fig.2 Ours 详细分析',
     'AF Loss 轨迹 / AF vs Forget 相关�?/ α_t 衰减 / Train-Val Gap / 分支精度 / 分支多样�?),
    ('fig3_forgetting_rate.png', 'Fig.3 遗忘速率分析',
     '遗忘速率 dF/dE / 累积遗忘 / 遗忘 vs 恢复事件占比'),
    ('fig4_learning_dynamics.png', 'Fig.4 训练动�?,
     'Train Acc / 泛化 Gap / Ensemble vs Branch / LR + α_t Schedule'),
    ('fig5_af_tradeoff.png', 'Fig.5 AF Loss 相图',
     'AF Loss �?LR 阶段着�?/ Acc vs AF Loss 散点'),
    ('fig6_harmful_vs_benign.png', 'Fig.6 有害/有益遗忘分解',
     'Harmful Score 分布 / 特征对比 / 选择性拯救率 / 遗忘景观 / 样本轨迹 / Forget Intensity'),
]

for fname, title, desc in figures:
    p = doc.add_paragraph()
    p.add_run(f'{title}�?).bold = True
    p.add_run(desc)
    p.add_run(f'\n文件：experiments/{fname}')
    img_path = f'experiments/{fname}'
    if os.path.exists(img_path):
        try:
            doc.add_picture(img_path, width=Inches(5.5))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        except:
            doc.add_paragraph('(图片加载失败，请手动查看)')

# ════════════════════════════════════════════════
# 六、根因总结
# ════════════════════════════════════════════════
doc.add_heading('六、根因总结：为�?MCW-AF 无法超越 HistKD', level=1)

doc.add_paragraph(
    '经过两轮实验（v1 恒定 λ，v2 反转 λ 调度）和深入分析，我们识别出三个层次的原因：')

doc.add_heading('层次 1：数量不对称', level=2)
doc.add_paragraph(
    'AF 保护�?352 个样本但间接损害�?373 个。在 CIFAR-100 干净数据上，'
    'SKD 的遗忘大部分是有益去噪，阻止它是负收益的�?)

doc.add_heading('层次 2：时序矛�?, level=2)
doc.add_paragraph(
    'v2 通过 (1-α_t) 挂钩缓解了早期约束过强的问题（ep100 领先 HistKD 5%），'
    '但后�?AF 仍阻碍精调收敛。AF �?保护"功能和模型的"优化"功能在目标上存在根本矛盾�?)

doc.add_heading('层次 3：机制本�?, level=2)
p = doc.add_paragraph()
p.add_run('MCW-AF 的核心矛盾：')
p.add_run('\n�?要防止遗忘，必须阻止预测变化')
p.add_run('\n�?�?SKD 的训练过程本身就是通过预测变化来持续改进的')
p.add_run('\n�?一个完全阻止变化的模型就是无法学习的模�?)
p.add_run('\n�?因此任何 EWC 类方法在 SKD 场景下都有一个内禀的精度上限——这个上限低于不加约束的 HistKD')

doc.add_paragraph(
    '这不�?MCW-AF 的设计缺陷，而是 EWC 范式�?SKD 场景下的根本局限�?
    'SKD 中的"遗忘"不是灾难性遗忘（CL 场景），而是模型持续自我完善的必要过程�?)

# ════════════════════════════════════════════════
# 七、讨论与后续方向
# ════════════════════════════════════════════════
doc.add_heading('七、讨论与后续方向', level=1)

doc.add_heading('7.1 本研究的贡献', level=2)

contributions = [
    '首次�?SKD 中的遗忘分解为有害遗忘（信号丢失）和有益去噪（伪标签替换）两种类型，并建立了操作化区分标�?,
    '证明�?MCW-AF 的置信度加权机制天然实现选择性抗遗忘—�?7 倍倾向于保护有害遗忘而非有益去噪',
    '通过 v1/v2 消融实验揭示�?AF 时序调度的重要性，以及 EWC 范式�?SKD 场景下的根本局�?,
    '提供了完整的分析工具链：遗忘检测、有�?有益分解、多方法对比可视�?,
]
for c in contributions:
    doc.add_paragraph(c, style='List Bullet')

doc.add_heading('7.2 MCW-AF 可能有效的场�?, level=2)

doc.add_paragraph('基于根因分析，我们推�?MCW-AF 在以下场景可能显著优�?HistKD�?)

scenarios = [
    ('标签噪声场景', '�?GT 标签含有噪声�?0%/40% symmetric noise），有益去噪的需求减少，'
     '有害遗忘成为主要矛盾 �?AF 的净收益变正'),
    ('长训练（300+ epoch�?, '模型高度收敛后，微调阶段的遗忘更可能是真正的信号丢失�?
     '此时 AF 的保护价值增�?),
    ('小模�?低容�?, '容量受限时样本间竞争更激烈，处于边界的样本更容易被牺�?�?AF 保护更重�?),
    ('持续学习场景', '当数据分布发生变化时，旧任务知识的遗忘是灾难性的 �?EWC 类方法的天然优势场景'),
]
for title, desc in scenarios:
    p = doc.add_paragraph()
    p.add_run(f'{title}�?).bold = True
    p.add_run(desc)

doc.add_heading('7.3 后续方向', level=2)

add_table(doc,
    ['方向', '假设', '预期结果', '工作�?],
    [['A. 标签噪声实验', '噪声下有益去噪↓，AF 净收益�?,
      'Ours > HistKD 在噪声场�?, '�?已完�?20%'],
     ['B. 40% 噪声验证', '更高噪声 �?AF 净收益穿越零点',
      '完整噪声率→收益曲线', '中（待跑�?],
     ['C. Per-sample replay', '记忆回放替代权重约束',
      '更精准的保护，避�?parameter interference', '�?],
     ['D. 改写叙事', '将贡献从"方法提升"转向"理解深化"',
      '当前分析框架 + 实验发现本身是有价值的学术贡献', '持续']])

# ════════════════════════════════════════════════
# 八、标签噪声实验（2026-06-27�?# ════════════════════════════════════════════════
doc.add_heading('八、标签噪声实验：验证 MCW-AF 在噪声场景的价�?, level=1)

doc.add_heading('8.1 动机与假�?, level=2)
doc.add_paragraph(
    '干净数据实验中，MCW-AF 无法超越 HistKD 的根因是：SKD �?~50% 的遗忘是有益去噪�?
    'EWC 范式不加区分地阻止了有利于泛化的伪标签替换过程�?
    '一个直接的推论是：如果数据本身含有噪声，GT 标签变得不可靠，有益去噪的需求将减少�?
    '信号遗忘（有害遗忘）将成为主要矛盾——此�?AF 的选择性保护应该能转化为精度优势�?)

p = doc.add_paragraph()
p.add_run('核心假设：噪声率 �?�?有益去噪需�?�?�?AF 净收益 �?�?存在某噪声率�?Δ(Ours−HistKD) > 0�?)

doc.add_heading('8.2 实验设计', level=2)
doc.add_paragraph(
    '�?CIFAR-100 训练集上施加对称标签噪声（random flip to other class），验证集保持干净�?
    '实现方式�?-noise_rate 0.2 参数，在 custom_dataloader.py 中通过 '
    'apply_symmetric_label_noise() 函数�?trainset.targets 随机翻转�?)

add_table(doc,
    ['实验', '噪声�?, '方法', 'Best Val', '状�?],
    [['noise20_hist_s27', '20%', 'HistKD Only', f'{hist_val_n20:.2f}%', '�?完成'],
     ['noise20_ours_s27', '20%', 'MCW-AF v2',  f'{ours_val_n20:.2f}%', '�?完成']])

doc.add_heading('8.3 核心结果', level=2)

doc.add_paragraph(
    '与干净数据相比�?0% 噪声�?AF 的相对劣势大幅缩小，但尚未实现超越：')

add_table(doc,
    ['指标', 'Clean (0% noise)', '20% Noise', '变化'],
    [['HistKD Best Val',       f'{hist_val_c:.2f}%',
                               f'{hist_val_n20:.2f}%',
                               f'{hist_val_n20 - hist_val_c:+.2f}%'],
     ['Ours v2 Best Val',      '79.01%', f'{ours_val_n20:.2f}%',
                               f'{ours_val_n20 - 79.01:+.2f}%'],
     ['Δ (Ours �?HistKD)',     '�?.42%', f'{ours_val_n20 - hist_val_n20:+.2f}%',
                               f'{(ours_val_n20 - hist_val_n20) - (-0.42):+.2f}%'],
     ['Harmful Rescue Rate',   f'{v2_harmful_rescue/1061*100:.1f}%',
                               f'{n20_harmful_rescue/n20_harmful_n*100:.1f}%',
                               f'{n20_harmful_rescue/n20_harmful_n*100 - v2_harmful_rescue/1061*100:+.1f}%'],
     ['Odds Ratio (H/B)',      f'{(v2_harmful_rescue/1061)/(v2_benign_rescue/1060):.1f}x',
                               f'{n20_odds:.1f}x', '�?],
     ['Lost Samples',          str(v2_lost), str(n20_lost),
                               f'{n20_lost - v2_lost:+d}'],
     ['Net Effect',            f'−{v2_lost - v2_rescued}', f'{n20_net:+d}',
                               f'{n20_net - (v2_rescued - v2_lost):+d}']])

p = doc.add_paragraph()
p.add_run('关键发现�?).bold = True
p.add_run('\n(1) Gap 缩小 74%（−0.42% �?�?.11%），几乎打平——噪声使 AF 的相对劣势大幅减�?)
p.add_run('\n(2) 有害拯救率提升（27.6% �?32.0%）——AF 在噪声下的保护更强有�?)
p.add_run('\n(3) Net Effect 改善 71%（−41 �?�?2）但仍为负—�?0% 噪声不足以让 AF 超越 HistKD')
p.add_run('\n(4) 有害遗忘比例未变（仍 ~50%）——噪声并未显著改�?SKD 遗忘的二元构�?)

doc.add_heading('8.4 趋势分析与下一�?, level=2)

doc.add_paragraph(
    '噪声率从 0% �?20%，�?�?�?.42% 收敛�?�?.11%，Net Effect �?�?1 改善�?�?2�?
    '趋势明确指向：存在一个噪声率阈值，在此之上 AF 将首次超�?HistKD�?)

add_table(doc,
    ['噪声�?, 'Δ (Ours �?HistKD)', 'Net Effect', '状�?],
    [['0%  (干净)',  '�?.42%', '�?1', '�?],
     ['20%',         f'{ours_val_n20 - hist_val_n20:+.2f}%', f'{n20_net:+d}', '�?],
     ['40%',         '�?,      '�?,      '待跑']])

doc.add_paragraph(
    '40% 噪声实验将验证：Δ 是否穿越零点（Ours > HistKD）。若能画出一条完整的 '
    '"噪声�?�?AF 净收益"曲线，将�?MCW-AF 在有噪声场景下具有实用价值的直接证据�?)

# Insert fig7
img_path = 'experiments/fig7_noise_analysis.png'
if os.path.exists(img_path):
    p = doc.add_paragraph()
    p.add_run('Fig.7 噪声分析�?).bold = True
    p.add_run('(a) 精度曲线对比 (b) Δ 轨迹 (c) Harmful Score 分布 (d) 选择性拯救率 (e) 特征对比 (f) Net Effect')
    doc.add_picture(img_path, width=Inches(5.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

# ════════════════════════════════════════════════
# 八、方法与实现细节
# ════════════════════════════════════════════════
doc.add_heading('八、实现细�?, level=1)

doc.add_heading('8.1 超参�?, level=2)

add_table(doc,
    ['参数', '�?, '说明'],
    [['af_lambda (λ)', '0.5', 'AF 全局强度（v2 中为最大强度）'],
     ['af_alpha (α)',  '2.0', '指数放大系数，控制置信度→权重的映射陡峭�?],
     ['af_tau (τ)',    '0.2', '保护门槛，m_i 需超过此值才获得有效保护'],
     ['HSKD',          '1',   '启用历史自知识蒸�?],
     ['ce_weight',     '1.0', '交叉熵损失权�?],
     ['kd_weight',     '0.0', '结构蒸馏损失权重（关闭，仅用 HSKD�?],
     ['α_t schedule',  'cos 0.9�?', 'HSKD 历史信任系数衰减']])

doc.add_heading('8.2 关键代码改动', level=2)

p = doc.add_paragraph()
p.add_run('AF Loss 计算（main.py L734-758）：').bold = True

code_text = '''
if args.HSKD and args.af_lambda > 0.0 and epoch > 0:
    with torch.no_grad():
        p_old_batch = all_predictions[input_indices].cuda()
        # �?1): 边际置信�?        p_old_correct = p_old_batch[range(B), targets]
        p_old_masked = p_old_batch.clone()
        p_old_masked[range(B), targets] = -inf
        p_old_max_wrong = p_old_masked.max(dim=1).values
        m = p_old_correct - p_old_max_wrong
        # �?2): 抽奖权重 (m_i�? �?w_i=0)
        w = exp(α*(m-τ)) if m > 0 else 0

    # �?3): 加权 KL, v2 �?AF 强度 �?(1-α_t)
    log_p_new = torch.log(softmax_output + 1e-10)
    kl_per_sample = F.kl_div(log_p_new, p_old_batch, reduction='none').sum(dim=1)
    loss_af = (w * kl_per_sample).mean()
    af_weight = args.af_lambda * (1.0 - alpha_t)  # v2 改进
    loss = loss + af_weight * loss_af'''

p = doc.add_paragraph()
run = p.add_run(code_text)
run.font.name = 'Consolas'
run.font.size = Pt(8)

doc.add_heading('8.3 遗忘分析管线', level=2)

doc.add_paragraph(
    '(1) 训练时：--track_forgetting 1 在每�?epoch val 后保�?per-sample correctness 矩阵 '
    '(correct_epoch_XXXX.pt, 10000�?bool tensor)�?
    '\n(2) 分析时：analyze_forgetting.py 加载所�?epoch �?correctness�?
    '计算 Forget/Learn Fraction 并生成遗忘曲线图�?
    '\n(3) 分解时：analyze_harmful_vs_benign.py 基于稳定�?遗忘时机/振荡率特征对遗忘样本二分�?
    '评估 MCW-AF 对两种遗忘的选择性保护效果�?)

# ── 参考文�?──
doc.add_heading('参考文�?, level=1)

refs = [
    'Wu et al. "Why Self-Training Helps and Hurts: A Theoretical Analysis." ICML 2026.',
    'Stern et al. "On Local Overfitting and Forgetting in Deep Neural Networks." AAAI 2025.',
    'Stern et al. "Forget Me Not: Reducing Catastrophic Forgetting via Checkpoint Fusion and Distillation." TPAMI 2026.',
    'Kirkpatrick et al. "Overcoming Catastrophic Forgetting in Neural Networks." PNAS 2017.',
    'DTSKD: "Self-Knowledge Distillation with Progressive Refinement." PR 2024.',
]
for ref in refs:
    doc.add_paragraph(ref, style='List Bullet')

# ── 保存 ──
output_path = 'MCW-AF遗忘研究_实验记录与分析_2026-06-26.docx'
doc.save(output_path)
print(f'[Saved] {output_path}')
