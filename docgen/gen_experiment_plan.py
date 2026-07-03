"""
生成后续实验计划 Word 文档
"""
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

doc = Document()

# ── 标题 ──
title = doc.add_heading('MCW-AF 后续实验计划', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph(
    '基于当前实验结果�?%/20%/50% 噪声�? seed，ResNet18-DTSKD，CIFAR-100），'
    '制定以下实验计划以完善论文的证据链�?,
    style='Normal'
)

# ── 辅助函数 ──
def set_cell_shading(cell, color):
    """设置单元格背景色"""
    shading_elm = cell._element.get_or_add_tcPr()
    shading = shading_elm.makeelement(qn('w:shd'), {
        qn('w:fill'): color,
        qn('w:val'): 'clear'
    })
    shading_elm.append(shading)

def add_styled_table(doc, headers, rows, col_widths=None):
    """创建带样式的表格"""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(9)

    # Data
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.rows[r + 1].cells[c]
            cell.text = str(val)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(9)

    return table


# ════════════════════════════════════════════════════════
# Phase 1: 统计验证
# ════════════════════════════════════════════════════════
doc.add_heading('Phase 1: 统计显著性验�?(P0 �?发表必要条件)', level=1)
doc.add_paragraph(
    '目的：当前所有实验仅使用 seed=27。需至少 3 �?seed 确认效果的统计显著性，'
    '并计�?error bar / 置信区间�?
)

add_styled_table(doc,
    ['#', '噪声�?, '方法', 'Seed', '预计 Best Val', '预计耗时', '备注'],
    [
        ['1', '0%',  'HistKD',   '42', '~79.4%', '~2h', '已有 CE s42 基线'],
        ['2', '0%',  'Ours v2',  '42', '~79.0%', '~2h', '验证 Δ 方向一致�?],
        ['3', '0%',  'HistKD',   '123','~79.4%', '~2h', ''],
        ['4', '0%',  'Ours v2',  '123','~79.0%', '~2h', ''],
        ['5', '20%', 'HistKD',   '42', '~75.6%', '~2h', ''],
        ['6', '20%', 'Ours v2',  '42', '~75.5%', '~2h', '验证 gap 缩小趋势'],
        ['7', '50%', 'HistKD',   '42', '~65.7%', '~2h', ''],
        ['8', '50%', 'Ours v2',  '42', '~66.0%', '~2h', '验证 AF 超越是否稳健'],
    ],
)

doc.add_paragraph(
    '预期产出�?-seed mean ± std 的噪声梯度图 (替换当前�?seed 版本)�?
    '对穿越点的误差估计。若 3 seed 结果一致，则具备统计说服力�?,
    style='Normal'
)

# ════════════════════════════════════════════════════════
# Phase 2: 密集噪声梯度
# ════════════════════════════════════════════════════════
doc.add_heading('Phase 2: 密集噪声梯度 �?精确定位穿越�?(P0)', level=1)
doc.add_paragraph(
    '目的：当前仅�?3 个噪声水�?(0%, 20%, 50%)，穿越点 ~28% 是线性插值估计�?
    '需补充中间噪声水平以绘制精确的穿越曲线。注意：需用原始训练配�?'
    '(batch_size=128, SGD)，而非�?batch LAMB 配置�?,
)

add_styled_table(doc,
    ['#', '噪声�?, '方法', 'Seed', '预计 Best Val', '备注'],
    [
        ['1',  '10%', 'HistKD',  '27', '~77.5%', ''],
        ['2',  '10%', 'Ours v2', '27', '~77.5%', '预测 Δ �?�?.25%'],
        ['3',  '30%', 'HistKD',  '27', '~70.5%', ''],
        ['4',  '30%', 'Ours v2', '27', '~70.5%', '预测 Δ �?0%, 穿越点附�?],
        ['5',  '40%', 'HistKD',  '27', '~68.0%', ''],
        ['6',  '40%', 'Ours v2', '27', '~68.2%', '预测 Δ �?+0.15%'],
    ],
)

doc.add_paragraph(
    '预期产出�? 点噪声梯度曲�?(0/10/20/30/40/50%)，精确穿越点位置�?
    '作为论文核心 Figure (替代当前�?3 点版�?�?,
    style='Normal'
)

# ════════════════════════════════════════════════════════
# Phase 3: 真实噪声数据�?# ════════════════════════════════════════════════════════
doc.add_heading('Phase 3: 真实噪声数据集验�?(P1)', level=1)
doc.add_paragraph(
    '目的：当前使用对称翻�?(symmetric flip) 模拟标签噪声�?
    '与真实世界噪声分布有差异。CIFAR-100N 提供 human-annotated 真实噪声标签�?
)

add_styled_table(doc,
    ['#', '数据�?, '噪声类型', '方法', 'Seed', '备注'],
    [
        ['1', 'CIFAR-100N (fine)', '人类标注噪声 ~40%', 'HistKD',  '27', ''],
        ['2', 'CIFAR-100N (fine)', '人类标注噪声 ~40%', 'Ours v2', '27', '预期 AF > HistKD'],
        ['3', 'CIFAR-100N (noisy)', '人类标注噪声 ~40%', 'HistKD',  '27', '不同噪声类型'],
        ['4', 'CIFAR-100N (noisy)', '人类标注噪声 ~40%', 'Ours v2', '27', '验证泛化�?],
    ],
)

doc.add_paragraph(
    '预期产出：真实噪声场景验证。若 AF 优势在真实噪声上更明显（因噪声分布更复杂），'
    '将极大增强论文的实践价值�?,
    style='Normal'
)

# ════════════════════════════════════════════════════════
# Phase 4: 架构泛化
# ════════════════════════════════════════════════════════
doc.add_heading('Phase 4: 架构泛化性验�?(P2)', level=1)
doc.add_paragraph(
    '目的：验�?MCW-AF 在不同网络架构上的有效性，排除�?ResNet 过拟合的可能�?
)

add_styled_table(doc,
    ['#', '架构', '噪声�?, '方法', 'Seed', '备注'],
    [
        ['1', 'WRN-28-10',  '0%',  'HistKD',  '27', '更宽的网�? 遗忘模式可能不同'],
        ['2', 'WRN-28-10',  '0%',  'Ours v2', '27', ''],
        ['3', 'WRN-28-10',  '50%', 'HistKD',  '27', ''],
        ['4', 'WRN-28-10',  '50%', 'Ours v2', '27', '验证噪声鲁棒性是否架构无�?],
        ['5', 'ViT-Tiny',   '0%',  'HistKD',  '27', 'Transformer vs CNN 遗忘差异'],
        ['6', 'ViT-Tiny',   '0%',  'Ours v2', '27', '需确认 DTSKD �?ViT 上的适配'],
        ['7', 'ViT-Tiny',   '50%', 'HistKD',  '27', ''],
        ['8', 'ViT-Tiny',   '50%', 'Ours v2', '27', ''],
    ],
)

# ════════════════════════════════════════════════════════
# Phase 5: 不对称噪�?# ════════════════════════════════════════════════════════
doc.add_heading('Phase 5: 不对称标签噪�?(P1)', level=1)
doc.add_paragraph(
    '目的：真实场景中噪声往往是不对称的（�?猫→�?�?猫→卡车"更常见）�?
    '验证 AF 在此类噪声下的有效性�?
)

add_styled_table(doc,
    ['#', '噪声类型', '噪声�?, '方法', '备注'],
    [
        ['1', 'Asymmetric (类内翻转)',  '20%', 'HistKD',  '更真实的噪声模式'],
        ['2', 'Asymmetric (类内翻转)',  '20%', 'Ours v2', ''],
        ['3', 'Asymmetric (类内翻转)',  '40%', 'HistKD',  ''],
        ['4', 'Asymmetric (类内翻转)',  '40%', 'Ours v2', '预测 gap 更大'],
    ],
)

# ════════════════════════════════════════════════════════
# Phase 6: 消融与对�?# ════════════════════════════════════════════════════════
doc.add_heading('Phase 6: 消融实验与方法对�?(P2)', level=1)

doc.add_heading('6.1 MCW-AF 组件消融', level=2)
add_styled_table(doc,
    ['#', '配置', '噪声�?, '目的'],
    [
        ['1', '�?stability gate (�?w_i × (1-α_t))', '50%', '消融 stability 的作�?],
        ['2', '�?(1-α_t) 调度 (�?w_i × stability)', '50%', '消融时序反转的作�?],
        ['3', '硬阈值替代指数权�?(w_i = 1 if m_i > τ else 0)', '50%', '消融指数加权的必要�?],
        ['4', '�?w_i (恒定 λ=0.5, �?stability, �?1-α_t)', '50%', 'v1 在噪声下的表�?],
    ],
)

doc.add_heading('6.2 �?SOTA 噪声鲁棒方法对比', level=2)
add_styled_table(doc,
    ['#', '方法', '噪声�?, '备注'],
    [
        ['1', 'Label Smoothing',              '50%', '标准噪声鲁棒 baseline'],
        ['2', 'MixUp',                         '50%', '数据增强方法'],
        ['3', 'Co-teaching (Han et al. 2018)', '50%', '经典噪声鲁棒训练方法'],
        ['4', 'DivideMix (Li et al. 2020)',    '50%', 'SOTA 噪声学习方法'],
        ['5', 'ELR (Liu et al. 2020)',         '50%', 'Early-Learning Regularization'],
    ],
)

doc.add_paragraph(
    '注：Phase 6 消融实验可使�?seed=27 �?seed 快速验证，'
    '方法对比需在统一框架下复现或引用原文结果�?,
    style='Normal'
)

# ════════════════════════════════════════════════════════
# Phase 7: 分析与写�?# ════════════════════════════════════════════════════════
doc.add_heading('Phase 7: 分析与论文写�?(并行)', level=1)
add_styled_table(doc,
    ['#', '任务', '产出', '依赖'],
    [
        ['1', '�?seed 综合分析脚本', '�?error bar 的噪声梯度图', 'Phase 1 + 2 完成'],
        ['2', '有害/有益遗忘在噪声下的变化分�?, '各噪声水平遗忘分解图', 'Phase 2 完成'],
        ['3', '绘制论文 Figure 1-6 终版', '论文级图�?, 'Phase 1-2 完成'],
        ['4', '撰写 Introduction + Related Work', '~1500 words', '可立即开�?],
        ['5', '撰写 Method', '~2000 words (已有中文�?', '可立即开�?],
        ['6', '撰写 Experiments', '~2500 words', 'Phase 1-2 完成'],
        ['7', '撰写 Discussion & Conclusion', '~1000 words', '全部实验完成'],
    ],
)

# ════════════════════════════════════════════════════════
# 汇�?# ════════════════════════════════════════════════════════
doc.add_heading('实验量汇�?, level=1)
add_styled_table(doc,
    ['Phase', '内容', '实验�?, '预计总耗时', '优先�?, '是否阻塞论文'],
    [
        ['1', '�?seed 验证',    '8',  '~16 GPU-h', 'P0', '�?�?无此无法发表'],
        ['2', '密集噪声梯度',    '6',  '~12 GPU-h', 'P0', '�?�?核心 Figure'],
        ['3', '真实噪声数据�?,  '4',  '~8 GPU-h',  'P1', '�?�?但大幅增强说服力'],
        ['4', '架构泛化',        '8',  '~16 GPU-h', 'P2', '�?�?可放�?Appendix'],
        ['5', '不对称噪�?,      '4',  '~8 GPU-h',  'P1', '�?],
        ['6', '消融与对�?,      '9',  '~18 GPU-h', 'P2', '�?],
        ['7', '论文写作',        '�?,  '�?,          'P0', '�?�?可与实验并行'],
    ],
)

doc.add_paragraph(
    '\n总计�?9 个训练实�?(~78 GPU-h)�? 写作。\n'
    '最小可行发表集 (MVP)：Phase 1 + 2 + 7 = 14 个实�?+ 论文初稿。\n'
    '强发表集：MVP + Phase 3 + 5 = 22 个实验。\n'
    '完整版：全部 39 个实验�?,
    style='Normal'
)

# ── 保存 ──
output_path = 'MCW-AF_后续实验计划.docx'
doc.save(output_path)
print(f'[Done] Saved to {output_path}')
