"""生成更新版开题报�? 基于边际置信度加权的抗隐性遗忘自知识蒸馏"""
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# 样式
style = doc.styles['Normal']
style.font.size = Pt(11)
style.font.name = 'Times New Roman'
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
style.paragraph_format.line_spacing = 1.5

for i in range(1, 4):
    hs = doc.styles[f'Heading {i}']
    hs.font.name = 'Times New Roman'
    hs.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

def add_body(doc, text, bold=False, indent=True):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    run = p.add_run(text)
    run.font.size = Pt(11)
    run.font.name = 'Times New Roman'
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    if bold:
        run.bold = True
    return p

def add_center(doc, text, size=12, bold=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.name = 'Times New Roman'
    if bold:
        run.bold = True
    return p

# ====== Cover ======
title = doc.add_heading('基于边际置信度加权的抗隐性遗忘自知识蒸馏', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_center(doc, '—�?开题报�?—�?, 14)

doc.add_paragraph('')
doc.add_paragraph('')

# ====== 1. 研究背景 ======
doc.add_heading('1. 研究背景与问�?, 1)

doc.add_heading('1.1 自知识蒸馏及其局限�?, 2)

add_body(doc, (
    '自知识蒸馏（Self-Knowledge Distillation, SKD）是一类无需外部教师模型的知识蒸馏范式，'
    '其核心思想是将模型自身在先前训练阶段的预测作为"历史教师"来指导当前阶段的训练�?
    '代表性工�?DTSKD（Dual Teachers for Self-Knowledge Distillation, PR 2024）通过维护历史预测'
    '矩阵并采用指数移动平均的方式将历史知识与结构知识融合，在 CIFAR-100 上取得了超越传统 KD 的性能�?
))

add_body(doc, (
    '然而，SKD 范式隐含着一个根本性风险：历史教师的预测分布可能包含错误或退化的知识�?
    '已有研究（Wu et al., ICML 2026）揭示了 SKD �?信号遗忘"�?噪声去噪"的权衡—�?
    '历史蒸馏在去除噪声的同时，也会错误地削弱正确的分类信号。这一问题在现�?SKD 文献�?
    '主要通过统一衰减系数 α_t 来被动缓解，缺乏样本级的主动干预机制�?
))

doc.add_heading('1.2 隐性遗忘：核心研究问题', 2)

add_body(doc, (
    '本研究进一步揭示了一个更深层但鲜有文献针对性建模的问题：即使在单一任务的常规迭代训练中�?
    '模型对同一批样本的判别置信度亦存在显著的非单调退化现象。具体而言，模型虽在后续轮次中仍保�?
    '分类正确（即 argmax p_new = y_i），但其决策边界已从高置信度区（�?p(y_i) > 0.9）滑向模糊区'
    '（如 p(y_i) �?0.3）。这�?隐性遗�?（Implicit Forgetting）——分类正确但置信度持续衰退—�?
    '是导致模型鲁棒性下降的潜在核心诱因。现�?SKD 方法由于对所有正确分类样本施加无差别的知识继�?
    '约束，无法有效区�?坚定正确的知�?�?正在退化的知识"，因而难以针对性地抑制隐性遗忘�?
), bold=True)

# ====== 2. 方法设计 ======
doc.add_heading('2. 方法设计', 1)

doc.add_heading('2.1 核心思想', 2)

add_body(doc, (
    '本研究的核心假设是：正确的分类方法不会给出模糊的答案，而是从底层样本差异中寻找到决定性的不同�?
    '基于此，我们提出一种基于边际置信度加权的抗隐性遗忘自知识蒸馏方法'
    '（Margin-Confidence Weighted Anti-Forgetting SKD, MCW-AF-SKD）�?
    '方法的核心设计包含三个机制：'
))

add_body(doc, (
    '�?）抽奖机制（Lottery Mechanism）：对于历史教师明显分错的样本（边际置信�?m_i �?0），'
    '完全放弃施加任何知识蒸馏约束，令模型通过标准交叉熵像随机抽奖一样重新探索正确的分类边界�?
    '这一机制确保错误知识不会被固化�?
))

add_body(doc, (
    '�?）指数型动态权重（Exponential Dynamic Weighting）：对于历史教师分对的样本，'
    '通过边际置信�?m_i = p_old(y_i) - max_{c≠y_i} p_old(c) 量化知识的可靠性，'
    '并采用指数函�?w_i = exp(α·(m_i-τ)) 将置信度连续映射为约束强度�?
    '高置信度样本获得指数级增强的保护（预防隐性遗忘），低置信度样本获得降�?
    '（避免固化模糊边界）。无硬阈值——权重随置信度平滑变化�?
))

add_body(doc, (
    '�?）前�?KL 锚定（Forward KL Anchoring）：采用 KL(p_old || p_new) 前向 KL 散度�?
    '要求新分布必须覆盖旧分布的所有高概率区域。这一方向选择确保正确分类方法的核心特�?
    '（旧分布中某个类别概率显著突出）在新分布中得以保持�?
))

doc.add_heading('2.2 数学形式', 2)

add_center(doc, '�?1) 边际置信�?  m_i = p_old(y_i) �?max_{c≠y_i} p_old(c)', bold=True)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run('m_i �?[�?, 1]。m_i > 0 表示教师正确分类，m_i �?0 表示教师错误分类�?).font.size = Pt(9)

add_center(doc, '�?2) 动态权�?  w_i = exp(α · (m_i �?τ)),  m_i > 0;  w_i = 0,  m_i �?0', bold=True)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run('α: 尺度因子; τ: 中性阈值。指数函数确保高置信度获得指数级增强保护�?).font.size = Pt(9)

add_center(doc, '�?3) 总损�?  L_total = L_CE + λ · (1/N) Σ_i w_i · KL(p_old || p_new)', bold=True)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run('λ: 约束强度。前�?KL 强制新分布继承旧分布的高概率区域�?).font.size = Pt(9)

doc.add_heading('2.3 �?EWC 的关�?, 2)

add_body(doc, (
    '本方法与弹性权重巩固（EWC, Kirkpatrick et al., PNAS 2017）共�?识别重要知识→差异化保护"'
    '的逻辑框架。根本区别在于操作空间：EWC 在参数空间通过 Fisher 信息矩阵实施约束，而本方法在输�?
    '分布空间通过边际置信度实施约束。后者的关键优势包括：（1）无需 Fisher 信息的额外前�?反向传播�?
    '�?）无需存储参数级重要性矩阵；�?）天然复�?SKD 已有的历史预测缓存；�?）样本级粒度而非参数级粒度�?
))

# ====== 3. 实验设计 ======
doc.add_heading('3. 实验设计', 1)

doc.add_heading('3.1 实验设置', 2)

add_body(doc, (
    '数据集：CIFAR-100。模型：ResNet-18 DTSKD 架构�? 输出头：backbone + 3 branches）�?
    '训练配置�?00 epoch，batch size 64，初始学习率 0.1（epoch 100/150 各衰�?10×），'
    'SGD optimizer，余�?α_t 调度�?.9�?.0）。所有实验使用随机种�?27�?
))

doc.add_heading('3.2 实验方案', 2)

add_body(doc, 'Phase 1: 遗忘基线测量�? 方法 × 1 seed�?00 epoch�?, bold=True)
add_body(doc, (
    '（a）CE Only：纯交叉熵训练，无任何蒸馏约束�?
    '（b）HistKD Only：DTSKD 的历史蒸馏分支（HSKD=1, ce_weight=1.0, kd_weight=0.0），无结构蒸馏�?
    '（c）MCW-AF-SKD (Ours)：在 HistKD 基础上增加置信度加权 KL 约束�?
), indent=False)

add_body(doc, 'Phase 2: 超参数筛选（7 组合 × 20 epoch 网格搜索�?, bold=True)
add_body(doc, (
    '�?α �?{1.0, 2.0, 4.0}、�?�?{0.1, 0.2, 0.4}、�?�?{0.1, 0.5, 1.0} 进行控制变量搜索�?
    '以验证精度和遗忘率为联合指标选择最优配置。默认设置：α=2.0, τ=0.2, λ=0.5�?
), indent=False)

add_body(doc, 'Phase 3: 完整训练与分析（最优配�?× 1 seed�?00 epoch�?, bold=True)
add_body(doc, (
    '使用 Phase 2 中确定的最优超参数运行完整 200 epoch 训练，分�?Forget Fraction�?
    'Learn Fraction、遗忘曲线、置信度分布变化等指标�?
), indent=False)

doc.add_heading('3.3 评估指标', 2)

add_body(doc, (
    '�?）Forget Fraction（Stern et al., AAAI 2025）：曾经正确→当前错误的样本比例�?
    '�?）Learn Fraction：曾经错误→当前正确的样本比例�?
    '�?）遗忘曲线：Forget Fraction �?epoch 的变化轨迹�?
    '�?）Val Accuracy：验证集 Top-1 精度（确保方法不损害最终性能）�?
    '�?）置信度退化率：高置信度正确样本在后续轮次中置信度下降的比例（隐性遗忘的直接度量）�?
), indent=False)

# ====== 4. 预期贡献 ======
doc.add_heading('4. 预期贡献', 1)

contributions = [
    '首次系统性地揭示并建�?SKD 中的"隐性遗�?现象——分类正确但置信度持续退化的问题�?,
    '提出 MCW-AF-SKD 方法：在输出分布空间通过边际置信度加权实现样本级自适应知识保护�?
    '完全规避了参数空间方法（�?EWC）的计算与存储开销�?,
    '设计"抽奖机制"——对错误知识果断放弃——与"指数型动态权�?——对正确知识差异化保护—�?
    '的协同框架，实现�?SKD 中信号保护与噪声抑制的精准平衡�?,
    '建立 SKD 与持续学习（CL）的跨领域连接，�?CL 社区的抗遗忘技术在 SKD 中的应用开辟新方向�?,
]

for i, c in enumerate(contributions):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0.74)
    run = p.add_run(f'（{i+1}）{c}')
    run.font.size = Pt(11)
    run.font.name = 'Times New Roman'
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

# ====== 5. 参考文�?======
doc.add_heading('5. 关键参考文�?, 1)

refs = [
    '[1] Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks." PNAS, 2017.',
    '[2] Wu et al. "Why Self-Training Helps and Hurts: A Unified Analysis." ICML, 2026.',
    '[3] Stern et al. "On Local Overfitting and Forgetting in Deep Neural Networks." AAAI, 2025.',
    '[4] Stern et al. "Forget Me Not: Reducing Catastrophic Forgetting in DNNs." TPAMI, 2026.',
    '[5] Xu et al. "Dual Teachers for Self-Knowledge Distillation." Pattern Recognition, 2024.',
    '[6] Hinton et al. "Distilling the Knowledge in a Neural Network." NeurIPS Workshop, 2015.',
]

for ref in refs:
    p = doc.add_paragraph(ref)
    p.paragraph_format.space_after = Pt(2)
    for run in p.runs:
        run.font.size = Pt(9.5)
        run.font.name = 'Times New Roman'

# Save
output_path = 'D:/Python_code/DTSKD/SKD遗忘问题_开题报�?docx'
doc.save(output_path)
print(f'开题报告已保存: {output_path}')
