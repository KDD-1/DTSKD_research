"""生成"基于边际置信度加权的抗遗忘自知识蒸馏"方法文档 Word �?""
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

doc = Document()

# ====== 全局样式设置 ======
style = doc.styles['Normal']
style.font.size = Pt(11)
style.font.name = 'Times New Roman'
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
style.paragraph_format.line_spacing = 1.5
style.paragraph_format.space_after = Pt(4)

# 设置标题样式的中文字�?for i in range(1, 4):
    heading_style = doc.styles[f'Heading {i}']
    heading_style.font.name = 'Times New Roman'
    heading_style.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

# ====== 辅助函数 ======
def add_math_paragraph(doc, text, bold=False, size=12, alignment=WD_ALIGN_PARAGRAPH.CENTER, italic=True):
    """添加数学公式段落 (使用斜体 Times New Roman 模拟)"""
    p = doc.add_paragraph()
    p.alignment = alignment
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.name = 'Times New Roman'
    run.italic = italic
    if bold:
        run.bold = True
    return p

def add_labeled_equation(doc, label, eq_text, size=12):
    """添加带编号的数学公式"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    # 公式主体
    run_eq = p.add_run(eq_text)
    run_eq.font.size = Pt(size)
    run_eq.font.name = 'Times New Roman'
    run_eq.italic = True
    # 编号
    run_label = p.add_run(f'    ({label})')
    run_label.font.size = Pt(size)
    run_label.font.name = 'Times New Roman'
    run_label.italic = False
    return p

def add_boxed_equation(doc, label, eq_text):
    """添加带框的公�?(用加�?大字号模�?"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(10)
    # 添加边框效果: 使用 paragraph border (通过XML)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    for edge in ['top', 'left', 'bottom', 'right']:
        border = OxmlElement(f'w:{edge}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:space'), '4')
        border.set(qn('w:color'), '333333')
        pBdr.append(border)
    pPr.append(pBdr)
    # 公式文本
    run = p.add_run(eq_text)
    run.font.size = Pt(12)
    run.font.name = 'Times New Roman'
    run.italic = True
    run.bold = True
    # 编号
    run_label = p.add_run(f'    ({label})')
    run_label.font.size = Pt(12)
    run_label.font.name = 'Times New Roman'
    run_label.italic = False
    run_label.bold = True
    return p

def add_formula_explanation(doc, items):
    """添加公式组件解释列表"""
    for item in items:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(2)

        # Bold part before colon
        if '�? in item:
            bold_part, rest = item.split('�?, 1)
            run_b = p.add_run(bold_part + '�?)
            run_b.bold = True
            run_b.font.size = Pt(10)
            run_b.font.name = 'Times New Roman'
            run_r = p.add_run(rest)
            run_r.font.size = Pt(10)
            run_r.font.name = 'Times New Roman'
        elif ':' in item:
            bold_part, rest = item.split(':', 1)
            run_b = p.add_run(bold_part + ':')
            run_b.bold = True
            run_b.font.size = Pt(10)
            run_b.font.name = 'Times New Roman'
            run_r = p.add_run(rest)
            run_r.font.size = Pt(10)
            run_r.font.name = 'Times New Roman'
        else:
            run_r = p.add_run(item)
            run_r.font.size = Pt(10)
            run_r.font.name = 'Times New Roman'

def add_body_text(doc, text, size=11):
    """添加正文段落,支持简单的 **bold** �?$latex$ 标记"""
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0.74)  # 两个字符缩进
    # 简单的解析: �?** �?$ 分割
    segments = []
    i = 0
    current = ''
    while i < len(text):
        if text[i:i+2] == '**':
            if current:
                segments.append(('normal', current))
                current = ''
            # 找到下一�?**
            j = text.find('**', i+2)
            if j == -1:
                current += text[i:]
                break
            segments.append(('bold', text[i+2:j]))
            i = j + 2
        elif text[i] == '$':
            if current:
                segments.append(('normal', current))
                current = ''
            # 找到下一�?$
            j = text.find('$', i+1)
            if j == -1:
                current += text[i:]
                break
            # 处理 LaTeX 标记
            latex = text[i+1:j]
            # 移除 \text{...} 包装
            latex = latex.replace('\\text{', '').replace('}', '')
            segments.append(('math', latex))
            i = j + 1
        else:
            current += text[i]
            i += 1
    if current:
        segments.append(('normal', current))

    if not segments:
        p.add_run(text).font.size = Pt(size)
        return p

    for seg_type, seg_text in segments:
        run = p.add_run(seg_text)
        run.font.size = Pt(size)
        run.font.name = 'Times New Roman'
        run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        if seg_type == 'bold':
            run.bold = True
        elif seg_type == 'math':
            run.italic = True
            run.font.name = 'Times New Roman'
    return p

def add_plain_para(doc, text, size=11, bold=False, alignment=None, indent=True):
    """添加纯文本段�?""
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    if alignment is not None:
        p.alignment = alignment
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.name = 'Times New Roman'
    run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    if bold:
        run.bold = True
    return p

def add_code_block(doc, code_text):
    """添加代码�?(灰色背景 + Consolas 字体)"""
    for line in code_text.strip().split('\n'):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.15
        # 添加灰色背景
        pPr = p._p.get_or_add_pPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:fill'), 'F0F0F0')
        shd.set(qn('w:val'), 'clear')
        pPr.append(shd)
        run = p.add_run(line if line else ' ')
        run.font.size = Pt(8.5)
        run.font.name = 'Consolas'

def make_table_word(doc, headers, rows, col_widths=None):
    """创建格式化表�?""
    table = doc.add_table(rows=len(rows)+1, cols=len(headers), style='Table Grid')
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 表头
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(9.5)
        run.font.name = 'Times New Roman'
        # 灰色表头背景
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:fill'), 'D9E2F3')
        shd.set(qn('w:val'), 'clear')
        tcPr.append(shd)

    # 数据�?    for r, row_data in enumerate(rows):
        for c, cell_text in enumerate(row_data):
            cell = table.rows[r+1].cells[c]
            cell.text = ''
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(cell_text)
            run.font.size = Pt(9.5)
            run.font.name = 'Times New Roman'

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)

    doc.add_paragraph('')  # table后空�?    return table

# ======================================================================
# 文档正式开�?# ======================================================================

# ====== Title ======
title = doc.add_heading('基于边际置信度加权的抗隐性遗忘自知识蒸馏', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('—�?方法推导与实�?—�?)
run.font.size = Pt(14)
run.font.name = 'Times New Roman'
run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
run.italic = True
run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

doc.add_paragraph('')

# ====== 1. 问题动机与方法概�?======
doc.add_heading('1. 问题动机与方法概�?, 1)

doc.add_heading('1.1 隐性遗忘：一个被忽视的根本问�?, 2)

overview_text = (
    '自知识蒸馏（Self-Knowledge Distillation, SKD）通过将模型上一轮训练的预测分布 p_old 作为'
    '"历史教师"来指导当前轮次的学习，已在多项工作中展现出显著的性能增益。然而，现有研究�?SKD �?
    '的遗忘现象仅关注了最浅层的一面——即�?p_old 分错类时，错误信号会代际传播导致"显性遗�?�?
)
add_body_text(doc, overview_text)

p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Cm(0.74)
run = p.add_run('本文揭示了一个更深层、但鲜有文献针对性建模的问题')
run.bold = True
run.font.size = Pt(11)
run.font.name = 'Times New Roman'
run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
rest = p.add_run(
    '：即使在单一任务的常规迭代训练中，模型对同一批样本的判别置信度亦存在显著的非单调退化现象�?
    '具体而言，模型虽在后续轮次中仍保持分类正确（�?argmax p_new = y_i），但其决策边界已从高置信度�?
    '（如 p(y_i) > 0.9）滑向模糊区（如 p(y_i) �?0.3）。这�?隐性遗�?（Implicit Forgetting）—�?
    '分类正确但置信度持续衰退——是导致模型鲁棒性下降的潜在核心诱因，但�?SKD 文献中几乎未被触及�?
)
rest.font.size = Pt(11)
rest.font.name = 'Times New Roman'
rest.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

doc.add_heading('1.2 方法设计哲学', 2)

overview_text2 = (
    '针对上述问题，本文提出一种基于边际置信度加权的抗隐性遗忘自知识蒸馏方法'
    '（Margin-Confidence Weighted Anti-Forgetting SKD, MCW-AF-SKD）�?
    '其核心设计包含三个层面：'
)
add_body_text(doc, overview_text2)

p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Cm(0.74)
run = p.add_run('�?�?抽奖机制"（Lottery Mechanism）——错误知识的果断放弃�?)
run.bold = True
run.font.size = Pt(11)
run.font.name = 'Times New Roman'
run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
rest = p.add_run(
    '对于历史教师明显分错的样本（边际置信�?m_i �?0），本方法完全放弃施加任何知识蒸馏约束，'
    '令模型通过标准交叉熵像随机抽奖一样重新探索正确的分类边界。这保证�?错误不被固化"�?
)
rest.font.size = Pt(11)
rest.font.name = 'Times New Roman'
rest.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Cm(0.74)
run = p.add_run('�?�?指数型动态权�?（Exponential Dynamic Weighting）——正确知识的差异化保护�?)
run.bold = True
run.font.size = Pt(11)
run.font.name = 'Times New Roman'
run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
rest = p.add_run(
    '对于历史教师分对的样本，核心挑战在于区分"坚定正确的知�?�?正在退化的知识"�?
    '高置信度样本（m_i 大，�?0.85）代表模型从底层特征中找到了决定性差异——这一类分类方法应当被强力锁定�?
    '低置信度样本（m_i 小，�?0.05）虽仍正确，但已处于退化边缘——过度保护反而会固化一个已经模糊的边界�?
    '因此，我们采用指数型权重 w_i = exp(α·(m_i-τ))，使得边际置信度的微小差异在高置信度区被放大为显著的'
    '约束强度差异，精准狙击隐性遗忘�?
)
rest.font.size = Pt(11)
rest.font.name = 'Times New Roman'
rest.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

p = doc.add_paragraph()
p.paragraph_format.first_line_indent = Cm(0.74)
run = p.add_run('�?�?前向 KL 锚定"（Forward KL Anchoring）——高概率区域的强制继承�?)
run.bold = True
run.font.size = Pt(11)
run.font.name = 'Times New Roman'
run.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
rest = p.add_run(
    '采用前向 KL 散度 KL(p_old || p_new)，要求新分布必须覆盖旧分布的所有高概率区域�?
    '这一选择的物理直觉是：正确的分类方法表现为旧分布中某个类别概率显著突出——前�?KL 确保这一突出'
    '在新分布中得以保持，从而防止决策边界在高置信度区发生漂移�?
)
rest.font.size = Pt(11)
rest.font.name = 'Times New Roman'
rest.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

overview_text3 = (
    '本方法与弹性权重巩固（EWC, Kirkpatrick et al., PNAS 2017）共�?识别重要知识→差异化保护"�?
    '逻辑框架。二者的根本区别在于：EWC 在参数空间通过 Fisher 信息矩阵衡量参数重要性，而本方法在输出分�?
    '空间通过边际置信度衡量知识可靠性。后者完全避免了 Fisher 信息的计算开销（无需额外前向-反向传播�?
    '与存储需求（无需维护参数级重要性矩阵），且天然复用 SKD 已有的历史预测缓存�?
)
add_body_text(doc, overview_text3)

# ====== 2. 数学推导 ======
doc.add_heading('2. 数学推导', 1)

# --- 2.1 问题形式�?---
doc.add_heading('2.1 问题形式�?, 2)

add_plain_para(doc,
    '设训练数据集 D = {(x_i, y_i)} (i = 1..N)，其�?y_i �?{1, 2, ..., C} 为真实标签�?
    '模型 f_θ: X �?R^C 输出 logits，经 softmax 归一化后得到类别概率分布�?, indent=True)

add_math_paragraph(doc,
    'p_new^(i) = softmax(f_θ(x_i)) �?[0,1]^C,   Σ_{c=1}^C p_new^(i)(c) = 1')

add_plain_para(doc,
    '记上一轮训练结束时的模型参数为 θ_old，对应的输出分布（历史教师）为：', indent=True)

add_math_paragraph(doc, 'p_old^(i) = softmax(f_{θ_old}(x_i))')

add_plain_para(doc, '标准 SKD 的总损失函数为�?, indent=True)

add_math_paragraph(doc,
    'L_SKD = L_CE(p_new^(i), y_i) + α_t · KL(p_old^(i) || p_new^(i))')

add_plain_para(doc,
    '其中 α_t 为随时间衰减的统一权重。关键问题：�?p_old^(i) 为错误分布时�?
    'KL(p_old || p_new) 的梯度会�?p_new 推向错误方向，造成遗忘�?, indent=True, bold=False)

# --- 2.2 边际置信�?---
doc.add_heading('2.2 边际置信度：识别"可靠知识"�?噪声"', 2)

add_plain_para(doc, '为区分上一轮预测的可靠性，定义边际置信度（Margin Confidence）：', indent=True)

add_labeled_equation(doc, '1',
    'm_i = p_old^(i)(y_i) �?max_{c≠y_i} p_old^(i)(c)')

add_plain_para(doc,
    '物理含义：m_i 度量了历史教师将真实类别 y_i 排在首位�?优势幅度"�?
    'm_i 的取值范围为 [�?, 1]�?, indent=True)

p = doc.add_paragraph(style='List Bullet')
run = p.add_run('m_i > 0�?)
run.bold = True
run.font.size = Pt(10)
run.font.name = 'Times New Roman'
p.add_run('教师正确分类。m_i 越接�?1，教师判断越坚定（真实类别的概率远高于其他类别）�?'
          '该样本的"历史知识"可靠，应当施�?KL 约束以保护之�?).font.size = Pt(10)

p = doc.add_paragraph(style='List Bullet')
run = p.add_run('m_i �?0�?)
run.bold = True
run.font.size = Pt(10)
run.font.name = 'Times New Roman'
p.add_run('教师错误分类。真实类别的概率被至少一个非真实类别超越 �?'
          '该样本的"历史知识"为噪声，应当完全放弃对其施加 KL 约束，避免错误代际传播�?).font.size = Pt(10)

add_plain_para(doc,
    '几何直觉：在 C 维概率单纯形（probability simplex）上，m_i 度量�?p_old^(i) 指向真实类别'
    '顶点 e_{y_i} �?净向量分量"——正值表示分布质量重心落在真实类别半空间内，负值表示已偏离'
    '至错误半空间�?, indent=True)

# --- 2.3 动态连续权重映�?---
doc.add_heading('2.3 动态连续权重映射：�?二元筛�?�?指数调节"', 2)

add_plain_para(doc,
    '对于 m_i > 0 的样本，并非以统一强度施加约束，而是通过指数型权重函数将边际置信度连续地映射'
    '为约束强度：', indent=True)

add_labeled_equation(doc, '2',
    'w_i = w(m_i) = exp(α · (m_i �?τ)),    m_i > 0')

add_plain_para(doc,
    '其中 α > 0 为尺度因子（Scale Factor），τ �?(0, 1) 为中性阈值（Neutral Threshold）�?, indent=True)

add_plain_para(doc, '�?(2) 的调节行为如下表所示：', indent=True, bold=True)

make_table_word(doc,
    ['条件', 'w_i 范围', '效应', '直觉'],
    [
        ['0 < m_i < τ', '(0, 1)', '降权', '教师分对了但不够自信 �?放松约束, 允许本轮微调'],
        ['m_i = τ', '1', '中�?, '恰好处于阈�?�?施加标准强度�?KL 约束'],
        ['m_i > τ', '(1, +�?', '提权', '教师高度自信地分�?�?加强约束, 锁定最优分类方�?],
    ],
    col_widths=[3.0, 2.0, 1.5, 9.5]
)

add_plain_para(doc,
    '选择指数函数的理由：�?）对 m_i 的变化高度敏感——边际置信度的微小提升在高置信度区域会被放大'
    '（指数函数的凸性），符�?越确定越要保�?的直觉；�?）非负性——权重始终为正，避免负权重导致的'
    '梯度反转问题；（3）单调递增——保�?更可靠的知识获得更强保护"的一致性�?, indent=True)

add_plain_para(doc,
    '对于 m_i �?0 的样本，直接�?w_i = 0，等价于�?KL 约束中完全移除该样本——此�?抽奖机制"�?
    '核心：让模型在该样本上仅依赖交叉熵损失自由探索，如同从随机初始状态出发重�?抽奖"一般，寻找'
    '正确的分类边界�?, indent=True)

# --- 2.4 总损失函�?---
doc.add_heading('2.4 总损失函�?, 2)

add_plain_para(doc,
    '将选择性加�?KL 散度与标准交叉熵结合，得到本方法的完整优化目标：', indent=True)

add_boxed_equation(doc, '3',
    'L_total = (1/N) Σ_i L_CE(p_new^(i), y_i)  +  λ · (1/N) Σ_i w_i · KL(p_old^(i) || p_new^(i))')
# 添加 underbrace 说明
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run('                 基础学习能力                                    置信度加权的抗遗忘约�?)
run.font.size = Pt(8)
run.font.name = 'Times New Roman'
run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

add_plain_para(doc, '�?(3) 的各组件详解�?, indent=True, bold=True)

components = [
    'L_CE(p_new, y) = −log p_new(y)：标准交叉熵损失，保证模型从真实标签中学习分类能力，是所有约束的基础�?,
    'KL(p_old || p_new) = Σ_c p_old(c) · log(p_old(c)/p_new(c))：前�?KL 散度（Forward KL Divergence），即用旧分布去"检�?新分布。选择此方向（而非反向 KL(p_new || p_old)）的理由是：前向 KL �?p_old(c) 大的区域�?p_new(c) 施加极强的惩罚——若旧分布在某类别上概率很高而新分布在该类别上概率很低，log(p_old/p_new) 项会爆涨，迫使新分布覆盖旧分布的所有高概率区域。这恰好契合"抗遗�?的需求：已掌握的分类知识（旧分布的高概率区域）必须被新分布继承。形象地讲，前向 KL 像一�?�?——用旧分布的高置信度区域死死拽住新分布，防止其在参数更新中漂移�?,
    'λ �?0：权衡系数，控制抗遗忘约束的整体强度。�?= 0 时退化为标准交叉熵训练；λ 过大时模型将被过度束缚在历史状态上，丧失学习新知识的能力�?,
    'w_i：式 (2) 定义的样本级动态权重，仅在 m_i > 0 时非零。注意当所�?w_i = 0 时（极端情况：上轮全错），KL 项整体退化为零，模型完全依赖交叉熵从头学习——这保证了方法在最坏情况下至少不会比标准训练更差�?,
]
add_formula_explanation(doc, components)

# --- 2.5 �?EWC 的形式化类比 ---
doc.add_heading('2.5 �?EWC 的形式化类比', 2)

add_plain_para(doc,
    '为阐明本方法的理论渊源，�?1 将其与经�?EWC 进行逐项对照�?, indent=True)

make_table_word(doc,
    ['维度', 'EWC (Kirkpatrick et al., 2017)', '本方�?(Ours)'],
    [
        ['约束空间', '参数空间 θ', '输出分布空间 p'],
        ['重要性度�?, 'Fisher 信息 F_i = E[(∂log p/∂θ_i)²]', '边际置信�?m_i'],
        ['约束形式', '(λ/2) Σ_i F_i(θ_i−θ_i*)²', 'w_i · KL(p_old || p_new)'],
        ['选择�?, '所有参数（无样本级区分�?, '仅对 m_i > 0 的样本施加约�?],
        ['重要性量化逻辑', '参数对似然的敏感�?, '历史教师对真实类别的确信�?],
        ['额外存储', 'θ* �?F（各 ~|θ| 维）', 'p_old（已�?SKD 自动维护�?],
    ],
    col_widths=[2.8, 5.8, 5.8]
)

add_plain_para(doc,
    '二者的核心共享洞察是：并非所有知识同等重要——重要的知识需要更强的保护。EWC 通过 Fisher 信息在参�?
    '空间实现这一点，本方法通过边际置信度在输出空间实现这一点。后者的优势在于无需额外计算和存储，天然�?'
    'SKD 的历史预测缓存机制适配�?, indent=True)

# --- 2.6 迭代更新与历史教师缓�?---
doc.add_heading('2.6 迭代更新与历史教师缓�?, 2)

add_plain_para(doc,
    '每一轮训练（epoch）结束后，执行历史教师缓存更新：', indent=True)

add_math_paragraph(doc,
    '∀i �?{1, ..., N}:   p_old^(i) �?p_new^(i) |_{θ_t}')

add_plain_para(doc,
    '即用当前轮次训练得到的模型参�?θ_t 重新对全体训练样本进行前向传播，将输出的概率分布保存为下一�?
    '�?p_old。这一步骤�?SKD 框架的标准操作，本方法不引入任何额外的前向传播开销——仅改变了历史分�?
    '被利用时的权重计算方式�?, indent=True)

# ====== 3. 训练伪代�?======
doc.add_heading('3. 训练伪代�?, 1)

add_plain_para(doc,
    '算法 1: 置信度加权抗遗忘自知识蒸�?(Confidence-Weighted Anti-Forgetting SKD)',
    bold=True, alignment=WD_ALIGN_PARAGRAPH.LEFT, indent=False)

code = r"""输入: D_train = {(x_i, y_i)}_{i=1}^N, 模型 f_θ,
       总训练轮�?E, 超参�?α (尺度因子), τ (中性阈�?, λ (约束强度)
输出: 训练完成的模�?f_θ

 1:  �?初始化历史预测缓�?(SKD 标准操作)
 2:  all_predictions �?[one_hot(y_i) for i = 1..N]
 3:
 4:  for epoch = 1 to E do
 5:      �?训练阶段
 6:      for each mini-batch (x_b, y_b) in D_train do
 7:          �?前向传播: 获取当前模型输出
 8:          logits_new �?f_θ(x_b)
 9:          p_new �?softmax(logits_new)
10:
11:          �?从缓存中读取历史分布
12:          p_old �?all_predictions[indices_of(x_b)]
13:
14:          �?计算标准交叉�?(基础学习能力)
15:          L_CE �?CrossEntropyLoss(logits_new, y_b)
16:
17:          �?计算样本级边际置信度 (�?)
18:          for i = 1 to |B| do
19:              p_old_correct �?p_old[i, y_b[i]]
20:              p_old_max_wrong �?max_{c≠y_b[i]} p_old[i, c]
21:              m_i �?p_old_correct - p_old_max_wrong
22:
23:              �?"抽奖机制": 分错的样本完全抛�?(不施加KL约束)
24:              if m_i �?0 then
25:                  w_i �?0
26:              else
27:                  �?动态权重映�?(�?)
28:                  w_i �?exp(α × (m_i - τ))
29:              end if
30:          end for
31:
32:          �?计算前向KL散度 (要求新分布覆盖旧分布的高概率区域)
33:          KL_per_sample �?KLDivLoss(p_old, p_new, reduction='none').sum(dim=1)
34:
35:          �?构建总损�?(�?)
36:          w �?tensor([w_1, ..., w_{|B|}])
37:          L_anti_forget �?mean(w × KL_per_sample)
38:          L_total �?L_CE + λ × L_anti_forget
39:
40:          �?反向传播与参数更�?41:          L_total.backward()
42:          optimizer.step()
43:          optimizer.zero_grad()
44:
45:          �?更新历史预测缓存 (与标�?HSKD 一�? 无额外开销)
46:          all_predictions[indices_of(x_b)] �?p_new.detach()
47:      end for
48:
49:      �?(可�? �?k 轮执行验�?50:      if epoch mod k_val == 0 then
51:          val_acc, forget_frac �?validate(f_θ, D_val, all_predictions)
52:          log(epoch, val_acc, forget_frac)
53:      end if
54:  end for
55:
56:  return f_θ"""

add_code_block(doc, code)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(8)
run = p.add_run('注：')
run.bold = True
run.font.size = Pt(9.5)
run.font.name = 'Times New Roman'
run.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
rest = p.add_run('算法�?2 行用 one-hot 标签初始化缓存等价于假设"�?0 轮教师完美分对所有样�?�?
                 '�?46 行的缓存更新与标�?HSKD 完全一致，不引入额外计算开销�?)
rest.font.size = Pt(9.5)
rest.font.name = 'Times New Roman'
rest.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

# ====== 4. 超参数讨�?======
doc.add_heading('4. 超参数讨�?, 1)

add_plain_para(doc,
    '本方法引入三个可调超参数：α（尺度因子）、τ（中性阈值）�?λ（约束强度）。以下基于常规图像分�?
    '任务（如 CIFAR-100 + ResNet-18）给出调参逻辑与取值建议�?, indent=True)

# --- 4.1 尺度因子 α ---
doc.add_heading('4.1 尺度因子 α', 2)

add_plain_para(doc,
    '作用：控制边际置信度到约束权重的"放大倍数"。�?越大，高置信度样本与低置信度样本之间的权重差距越悬殊�?,
    indent=True, bold=False)

add_plain_para(doc, '取值建议：', indent=True, bold=True)

make_table_word(doc,
    ['α 范围', '效应', '适用场景'],
    [
        ['[0.5, 1.0]', '温和调节：权重在 [e^{-0.5τ}, e^{0.5(1−�?}] 范围内平滑变�?,
         '教师质量一般、边际置信度分布分散�?],
        ['[1.0, 3.0]', '推荐区间：显著区分高/低置信度样本', '常规分类任务，教师准确率 > 60% �?],
        ['[3.0, 5.0]', '激进调节：高置信度样本获得极强约束，几乎无法被改变',
         '教师质量很高、仅需微调�?],
    ],
    col_widths=[2.5, 7.0, 5.0]
)

add_plain_para(doc,
    '调参策略：建议从 α = 2.0 起步。若观察到遗忘率下降但验证精度同步下降（过度约束），适当减小 α�?
    '若遗忘率下降不明显，适当增大 α。可在训练过程中监控 w_i 的均值与方差以判断权重分布是否合理——均�?
    '过小说明大部分样本被弱化，均值过大说明少数样本过度主导约束�?, indent=True)

# --- 4.2 中性阈�?τ ---
doc.add_heading('4.2 中性阈�?τ', 2)

add_plain_para(doc,
    '作用：定�?标准强度"的基准线。m_i = τ 时，w_i = 1（KL 约束强度等同于不加权的标�?SKD）�?,
    indent=True, bold=False)

add_plain_para(doc, '取值建议：', indent=True, bold=True)

make_table_word(doc,
    ['τ 范围', '效应', '适用场景'],
    [
        ['[0.05, 0.15]', '"宽松�?：仅有极低置信度的正确样本会被降权，绝大多数样本获得提权',
         '教师性能优异、期望大力保护时'],
        ['[0.15, 0.30]', '推荐区间：适中的分界线', '常规任务'],
        ['[0.30, 0.50]', '"严苛�?：需要较高的边际置信度才能获得提�?,
         '教师噪声较大、期望谨慎继承时'],
    ],
    col_widths=[2.5, 7.0, 5.0]
)

add_plain_para(doc,
    '调参策略：建议从 τ = 0.2 起步。可在训练早期观察被降权�? < m_i < τ）和被提权（m_i > τ）的样本'
    '比例，理想情况下应大致相当。若被降权的样本比例超过 70%，说�?τ 偏高，应适当降低；若被提权样本比�?
    '超过 80%，说�?τ 偏低，教师的大部分判断尚未达�?值得保护"的门槛就已获得增强约束�?, indent=True)

# --- 4.3 约束强度 λ ---
doc.add_heading('4.3 约束强度 λ', 2)

add_plain_para(doc,
    '作用：控制抗遗忘约束项在总损失中的相对权重。�?= 0 退化为标准交叉熵训练（无遗忘保护）；�?过大�?
    '模型创新不足�?, indent=True, bold=False)

add_plain_para(doc, '取值建议：', indent=True, bold=True)

make_table_word(doc,
    ['λ 范围', '效应', '适用场景'],
    [
        ['[0.01, 0.1]', '弱约束：仅对极高置信度样本产生微弱保�?, '教师变化剧烈、约束需谨慎�?],
        ['[0.1, 1.0]', '推荐区间：在遗忘抑制与学习灵活性之间取得平�?, '常规分类任务'],
        ['[1.0, 5.0]', '强约束：强力锁定已掌握的知识', '遗忘问题严重、或训练后期仅需精调�?],
    ],
    col_widths=[2.5, 7.0, 5.0]
)

add_plain_para(doc,
    '调参策略：建议从 λ = 0.1 起步，观察遗忘曲线（Forget Fraction vs. Epoch）与验证精度曲线的相�?
    '关系。理想状态下，遗忘率相比基线（如 CE Only 或标�?HSKD）降�?10%�?0%，而最终验证精度不下降'
    '或仅微弱下降�?0.5%）。若验证精度显著下降�?1%），降低 λ；若遗忘率未得到有效抑制，提�?λ�?,
    indent=True)

# --- 4.4 联合调参流程 ---
doc.add_heading('4.4 联合调参流程（推荐）', 2)

tuning_code = """1. 固定 α = 2.0, τ = 0.2, λ = 0.1 (默认设置)
2. 运行 30-50 轮探索性训�?
   a. 监控 w_i 分布 �?调节 τ (确保权重�?[0.1, 5.0] 范围内覆盖多数样�?
   b. 监控 Forget Fraction �?调节 λ (确保遗忘率下�?�?10%)
3. 细调 α: 若高置信�?(m_i > 0.5) 样本仍出现遗�?�?增大 α
4. 细调 λ: 在验证精度可接受的前提下, 最大化 λ 以抑制遗�?5. 最终运行完整训�?(300 epoch) 并报告结�?""
add_code_block(doc, tuning_code)

# ====== 5. 方法特性总结 ======
doc.add_heading('5. 方法特性总结', 1)

make_table_word(doc,
    ['特�?, '说明'],
    [
        ['选择�?, '仅对 m_i > 0 的样本施�?KL 约束，错误知识不被继�?],
        ['连续�?, '指数型权重实现置信度→约束强度的平滑映射，无硬阈值断�?],
        ['无额外开销', '所有计算均复用 SKD 已有的历史预测缓存和前向传播'],
        ['方向正确�?, '前向 KL(p_old || p_new) 保证新分布被旧分布的高概率区�?锚定"'],
        ['理论根基', '�?EWC 共享"重要性加权的知识保护"范式，但在更自然的输出空间操�?],
        ['安全保障', '最坏情况（上轮全错）退化为标准交叉熵，不会比基线更�?],
        ['可插拔�?, '可独立使用（替代 HSKD），也可�?StructKD、EWC-SKD 等组�?],
    ],
    col_widths=[3.0, 13.0]
)

# ====== Footer ======
doc.add_paragraph('')
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run(f'—�?文档版本 v1.0，生成于 2026-06-25 —�?)
run.font.size = Pt(9)
run.font.name = 'Times New Roman'
run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
run.italic = True

# ====== Save ======
output_path = 'D:/Python_code/DTSKD/方法推导与实现_置信度加权抗遗忘SKD.docx'
doc.save(output_path)
print(f'Word 文档已保�? {output_path}')
