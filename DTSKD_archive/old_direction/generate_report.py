"""
生成开题报告 Word 文档
用法: python generate_report.py
输出: D:\Python_code\DTSKD\DTSKD噪声鲁棒性研究_开题报告.docx
"""
from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import datetime


def set_cell_shading(cell, color):
    """Set cell background color"""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def set_cell_border(cell, **kwargs):
    """Set cell borders"""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}></w:tcBorders>')
    for edge, val in kwargs.items():
        element = parse_xml(
            f'<w:{edge} {nsdecls("w")} w:val="{val.get("val", "single")}" '
            f'w:sz="{val.get("sz", 4)}" w:space="0" w:color="{val.get("color", "000000")}"/>'
        )
        tcBorders.append(element)
    tcPr.append(tcBorders)


def add_formatted_paragraph(doc, text, style_name=None, font_name='仿宋', font_size=12,
                            bold=False, alignment=None, space_after=6, space_before=0,
                            first_line_indent=None, color=None, italic=False):
    """Add a paragraph with full formatting control"""
    if style_name:
        p = doc.add_paragraph(style=style_name)
    else:
        p = doc.add_paragraph()

    if alignment is not None:
        p.alignment = alignment

    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    if first_line_indent:
        pf.first_line_indent = Cm(first_line_indent)

    run = p.add_run(text)
    run.font.name = font_name
    run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name)
    run.font.size = Pt(font_size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)

    return p


def add_heading_styled(doc, text, level=1):
    """Add a heading with proper Chinese font"""
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = '黑体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        if level == 1:
            run.font.size = Pt(16)
        elif level == 2:
            run.font.size = Pt(14)
        elif level == 3:
            run.font.size = Pt(13)
    return h


def add_body_text(doc, text, first_line_indent=True):
    """Add body text paragraph in 仿宋"""
    return add_formatted_paragraph(
        doc, text, font_name='仿宋', font_size=12,
        alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
        first_line_indent=0.74 if first_line_indent else None,
        space_after=4, space_before=2
    )


def format_table(table, header_color="1F4E79", header_font_color=(255, 255, 255)):
    """Apply professional formatting to a table"""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Format header row
    for cell in table.rows[0].cells:
        set_cell_shading(cell, header_color)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = '黑体'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
                run.font.size = Pt(10)
                run.bold = True
                run.font.color.rgb = RGBColor(*header_font_color)

    # Format data rows
    for i, row in enumerate(table.rows[1:], 1):
        bg = "F2F7FB" if i % 2 == 0 else "FFFFFF"
        for cell in row.cells:
            set_cell_shading(cell, bg)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.name = '仿宋'
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋')
                    run.font.size = Pt(9)


def add_styled_table(doc, headers, rows, col_widths=None):
    """Create a professionally styled table"""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'

    # Header
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ''
        p = cell.paragraphs[0]
        run = p.add_run(header)
        run.font.name = '黑体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        run.font.size = Pt(10)
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_shading(cell, "1F4E79")

    # Data rows
    for i, row_data in enumerate(rows):
        bg = "F2F7FB" if i % 2 == 0 else "FFFFFF"
        for j, cell_text in enumerate(row_data):
            cell = table.rows[i + 1].cells[j]
            cell.text = ''
            p = cell.paragraphs[0]
            run = p.add_run(str(cell_text))
            run.font.name = '仿宋'
            run._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋')
            run.font.size = Pt(9)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_cell_shading(cell, bg)

    if col_widths:
        for i, width in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(width)

    doc.add_paragraph()  # spacer
    return table


def create_report():
    doc = Document()

    # ============================================================
    # Page setup
    # ============================================================
    section = doc.sections[0]
    section.page_width = Cm(21.0)    # A4
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.17)
    section.right_margin = Cm(3.17)

    # Set default font
    style = doc.styles['Normal']
    font = style.font
    font.name = '仿宋'
    font.size = Pt(12)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋')

    # ============================================================
    # COVER PAGE
    # ============================================================
    # Spacing
    for _ in range(6):
        doc.add_paragraph()

    # Title
    title_lines = [
        "开 题 报 告",
    ]
    for line in title_lines:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(line)
        run.font.name = '黑体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        run.font.size = Pt(26)
        run.bold = True
        p.paragraph_format.space_after = Pt(12)

    doc.add_paragraph()

    # Subtitle
    sub_title = "自知识蒸馏在非理想数据条件下的鲁棒性研究"
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(sub_title)
    run.font.name = '黑体'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    run.font.size = Pt(18)
    run.bold = True
    p.paragraph_format.space_after = Pt(8)

    doc.add_paragraph()

    # English sub-title
    en_title = "—基于 DTSKD (Dual Teachers for Self-Knowledge Distillation) 框架的噪声鲁棒性分析—"
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(en_title)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.italic = True
    p.paragraph_format.space_after = Pt(40)

    for _ in range(4):
        doc.add_paragraph()

    # Info table on cover
    info_items = [
        ("研究方向", "机器学习 · 知识蒸馏 · 鲁棒性分析"),
        ("实验平台", "PyTorch 2.4 + CIFAR-100 + ResNet18-DTSKD"),
        ("日　　期", datetime.date.today().strftime("%Y年%m月%d日")),
    ]
    for label, value in info_items:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(f"{label}：{value}")
        run.font.name = '仿宋'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋')
        run.font.size = Pt(14)
        p.paragraph_format.space_after = Pt(8)

    # Page break
    doc.add_page_break()

    # ============================================================
    # ABSTRACT
    # ============================================================
    add_heading_styled(doc, "摘  要", level=1)

    abstract_text = (
        "自知识蒸馏（Self-Knowledge Distillation, SKD）通过模型自身的历史预测作为软监督信号，"
        "在图像分类等任务上取得了显著的性能提升。DTSKD (Li et al., PR 2024) 提出了双教师机制——"
        "历史教师（上一轮 epoch 的预测）与结构教师（当前 backbone 输出）——将自知识蒸馏的性能推向新高度。"
        "已有理论工作 (Das & Sanghavi, ICML 2023) 分析了标准自蒸馏在标签噪声下的行为，"
        "多种噪声鲁棒 SKD 方法 (GEIKD, NoRD, SNTD) 也已被提出。然而，针对 DTSKD 特有的双教师架构"
        "——特别是其动态预测矩阵反馈循环——在非理想数据条件下的系统消融分析，目前仍为空白。"
    )
    add_body_text(doc, abstract_text)

    abstract_text2 = (
        "本研究旨在填补这一空白。我们提出假设：DTSKD 的历史教师机制建立在「教师可靠」的隐含前提之上；"
        "当训练数据存在标签噪声或输入损坏时，历史教师可能从知识来源转变为错误传播器，通过预测矩阵的反馈循环"
        "放大噪声，导致模型性能退化速度超过普通交叉熵训练。为验证该假设，我们设计了系统的消融实验，"
        "在受控的标签噪声（0%-40%）和输入损坏（高斯噪声/模糊/椒盐噪声）条件下，对比 CE Only、"
        "HistKD Only、StructKD Only 和 Full DTSKD 四种训练范式的性能退化行为。"
        "本研究不提出新方法，而是通过严谨的实验分析，揭示自知识蒸馏各组件的噪声敏感性，"
        "为后续鲁棒 SKD 方法的设计提供依据。"
    )
    add_body_text(doc, abstract_text2)

    add_body_text(doc, "关键词：自知识蒸馏；标签噪声；鲁棒性分析；DTSKD；历史教师；消融实验")

    doc.add_page_break()

    # ============================================================
    # TABLE OF CONTENTS (manual)
    # ============================================================
    add_heading_styled(doc, "目  录", level=1)
    toc_items = [
        ("一、研究背景与问题提出", 3),
        ("　　1.1 自知识蒸馏的成功", 3),
        ("　　1.2 被忽视的问题：教师可靠性的隐含假设", 3),
        ("　　1.3 研究问题与意义", 4),
        ("二、文献综述与创新性定位", 5),
        ("　　2.1 自知识蒸馏方法谱系", 5),
        ("　　2.2 自知识蒸馏在标签噪声下的理论分析（最相关工作）", 5),
        ("　　2.3 噪声鲁棒自蒸馏方法（竞争性方法）", 6),
        ("　　2.4 知识蒸馏中的不确定度传播", 7),
        ("　　2.5 研究空白与创新性定位", 7),
        ("三、研究假设", 8),
        ("　　3.1 主假设", 8),
        ("　　3.2 子假设", 8),
        ("　　3.3 竞争假设", 8),
        ("四、实验设计", 9),
        ("　　4.1 实验框架", 9),
        ("　　4.2 实验因子", 9),
        ("　　4.3 固定控制变量", 10),
        ("　　4.4 噪声注入技术方案", 10),
        ("　　4.5 评估指标", 11),
        ("　　4.6 实验矩阵", 11),
        ("　　4.7 统计考虑", 12),
        ("五、预期结果与解释框架", 13),
        ("　　5.1 结果-解释映射表", 13),
        ("　　5.2 论文潜力评估", 13),
        ("六、实施计划", 14),
        ("七、贡献陈述", 15),
        ("参考文献", 16),
        ("附录 A：已完成的前期工作", 17),
        ("附录 B：实验命名规范", 17),
    ]
    for item, page in toc_items:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(14.5), alignment=WD_ALIGN_PARAGRAPH.RIGHT, leader=1)
        run = p.add_run(f"{item}\t{page}")
        run.font.name = '仿宋'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋')
        run.font.size = Pt(12)

    doc.add_page_break()

    # ============================================================
    # CHAPTER 1: 研究背景与问题提出
    # ============================================================
    add_heading_styled(doc, "一、研究背景与问题提出", level=1)

    add_heading_styled(doc, "1.1 自知识蒸馏的成功", level=2)
    add_body_text(doc,
        "知识蒸馏（Knowledge Distillation, KD）通过在教师-学生框架中迁移知识，在模型压缩和性能提升方面"
        "取得了显著成功。传统 KD 依赖一个独立的大型预训练教师网络，而自知识蒸馏（Self-Knowledge Distillation, SKD）"
        "消除了对外部教师的依赖——学生网络从自身的训练过程中提取知识。"
    )
    add_body_text(doc,
        "DTSKD (Li et al., Pattern Recognition 2024) 是 SKD 的代表性工作，提出了双教师机制："
        "历史教师（History Teacher）使用上一轮 epoch 的模型预测作为软标签，提供时序一致性监督；"
        "结构教师（Structural Teacher）使用 backbone 主干的输出指导各辅助分支，提供结构一致性监督。"
        "其核心公式为 soft_target = α_t · y_one-hot + (1-α_t) · ŷ_prev，"
        "其中 α_t 随训练从 0.9 余弦衰减至 0.0，控制真实标签与历史预测的混合比例。"
    )

    add_heading_styled(doc, "1.2 被忽视的问题：教师可靠性的隐含假设", level=2)
    add_body_text(doc,
        "DTSKD 的 History Teacher 机制建立在一个隐含前提之上："
        "上一轮模型的预测是可靠的，可以作为下一轮训练的有效监督信号。"
        "然而在现实场景中，这一前提并不总是成立。当训练数据存在标签噪声时，模型被迫拟合错误标签，"
        "导致预测质量下降，进而使历史教师传播错误；当输入数据被损坏时，模型从损坏图像中提取特征，"
        "预测不确定性增加，历史教师的质量也随之波动。"
    )

    add_body_text(doc,
        "更重要的是，DTSKD 的历史教师机制包含一个潜在的反馈循环（Feedback Loop）："
        "模型在当前 epoch 产生的预测被存储到预测矩阵 all_predictions 中，"
        "下一轮作为软目标的一部分用于训练。如果这些预测本身已被噪声污染，"
        "这种「自举式」（bootstrapping）机制可能导致错误在训练过程中逐轮累积放大。"
    )

    add_heading_styled(doc, "1.3 研究问题与意义", level=2)
    add_body_text(doc,
        "核心研究问题：当训练数据存在标签噪声或输入损坏时，DTSKD 的历史教师是否会从「知识来源」变为"
        "「错误传播器」，导致模型性能退化速度超过普通交叉熵训练？"
    )
    add_body_text(doc,
        "若该假设成立，意味着 DTSKD 在实际部署中（往往存在标注噪声）可能不如论文报告的那样有效，"
        "需要设计「教师可靠性感知」的机制来修正这一问题。若假设不成立（DTSKD 在噪声下反而更鲁棒），"
        "则意味着历史教师的 EMA 式软目标可能提供了隐式的标签平滑正则化，"
        "多分支结构蒸馏可能增强了模型的抗噪能力，这将为 DTSKD 提供一个新的优势论证维度。"
    )

    doc.add_page_break()

    # ============================================================
    # CHAPTER 2: 文献综述与创新性定位
    # ============================================================
    add_heading_styled(doc, "二、文献综述与创新性定位", level=1)

    add_heading_styled(doc, "2.1 自知识蒸馏方法谱系", level=2)
    add_body_text(doc,
        "自知识蒸馏方法在近年取得了显著进展。从 Born-Again Networks (ICLR 2018) 的多代自蒸馏，"
        "到 PS-KD (ICCV 2021) 的渐进式软目标混合，再到 EEKD (2022) 的历史中间模型集成蒸馏，"
        "研究者不断探索更有效利用模型自身知识的方式。DTSKD (PR 2024) 通过同时使用历史教师和结构教师，"
        "代表了该方向的最新进展。近期工作如 DimSelfKD (2025) 从维度视角蒸馏历史知识，"
        "DPSD (2024) 提出时维解耦与知识校准以解决历史输出低置信度问题。"
    )

    add_styled_table(doc,
        ["方法", "年份", "会议/期刊", "核心机制", "教师来源"],
        [
            ["Born-Again Networks", "2018", "ICLR", "多代自蒸馏", "上一代模型"],
            ["PS-KD", "2021", "ICCV", "渐进式软目标混合", "上一 epoch 自身预测"],
            ["CS-KD", "2021", "AAAI", "标准化约束自正则化", "当前模型 softmax"],
            ["EEKD", "2022", "arXiv", "历史中间模型集成蒸馏", "训练中多个 checkpoint"],
            ["DLB", "2022", "CVPR", "上一 mini-batch 一致性正则化", "上一 mini-batch 预测"],
            ["GEIKD", "2023", "CVIU", "门控集成自教师 + 影响函数去噪", "门控融合多分辨率特征"],
            ["DimSelfKD", "2025", "Sci. China Inf. Sci.", "维度历史知识蒸馏", "历史样本队列特征"],
            ["DTSKD", "2024", "PR", "双教师（历史+结构）", "上一 epoch 预测 + backbone"],
        ],
        col_widths=[2.8, 1.0, 2.5, 4.0, 4.5]
    )

    # ---- Section 2.2 ----
    add_heading_styled(doc, "2.2 自知识蒸馏在标签噪声下的行为——已有研究基础", level=2)

    add_body_text(doc,
        "一个重要的事实是：「自知识蒸馏在标签噪声下的鲁棒性」这一一般性问题，已经有多项工作进行了研究。"
        "本节系统综述这些已有成果，并在此基础上明确本研究的差异化定位。"
    )

    add_body_text(doc, "（1）Das & Sanghavi (ICML 2023) — 理论奠基", first_line_indent=False)
    add_body_text(doc,
        "Das & Sanghavi 在论文「Understanding Self-Distillation in the Presence of Label Noise」"
        "(ICML 2023, PMLR 202:7102-7140) 中对标准自蒸馏在标签噪声下的行为进行了严格的理论分析。"
        "该文分析的自蒸馏软目标公式 ξ · teacher_prediction + (1-ξ) · noisy_label 在数学上与 DTSKD 的 "
        "α_t · one_hot + (1-α_t) · last_prediction 等价。关键发现包括：在高标签噪声下，"
        "最优 ξ 可以大于 1（即「反学习」噪声标签）；自蒸馏引入了 bias-variance tradeoff——"
        "增大 ξ 减少方差但增加偏差；首次证明了交叉熵损失下自蒸馏在标签噪声中具有理论收益。"
        "该论文是目前该方向最重要的理论基准。"
    )

    add_body_text(doc, "（2）Chung et al. (2024/2025) — 标签平均视角", first_line_indent=False)
    add_body_text(doc,
        "Chung 等在「Rethinking Self-Distillation: Label Averaging and Enhanced Soft Label Refinement "
        "with Partial Labels」(arXiv:2402.10482) 中揭示了自蒸馏的本质：在特征空间中等效于标签平均"
        "（Label Averaging）——在特征高度相关的样本之间平滑标签。该文进一步证明多轮 SD 存在收益递减，"
        "在特定噪声条件下单轮 Partial Label Learning 学生模型反而更优，"
        "并给出了达到 100% 准确率所需的标签损坏条件和最小蒸馏轮数。"
    )

    add_body_text(doc, "（3）Wu et al. (2025) — 去噪与信号遗忘的权衡", first_line_indent=False)
    add_body_text(doc,
        "Wu 等在「Why Self-Training Helps and Hurts: Denoising vs. Signal Forgetting」"
        "(arXiv:2602.14029) 中从高维渐近理论角度分析了迭代自训练中的根本性权衡："
        "随机误差（噪声）随迭代次数快速衰减，但系统性误差（信号）也随迭代累积——"
        "两者相互作用产生 U 型测试风险曲线，存在最优早停时间。"
        "该文提出了迭代广义交叉验证准则用于数据驱动地选择停止时刻，"
        "从理论上解释了为什么自蒸馏在适度使用时有帮助，但过度使用会有害。"
    )

    add_body_text(doc, "（4）GEIKD (Liu et al., CVIU 2023) — DTSKD 团队的噪声鲁棒方法", first_line_indent=False)
    add_body_text(doc,
        "值得特别注意的是，DTSKD 论文的合著者（Zheng Li, Zhigeng Pan）在 CVIU 2023 上发表了 GEIKD "
        "（「GEIKD: Self-Knowledge Distillation Based on Gated Ensemble Networks and Influences-Based "
        "Label Noise Removal」）。该工作明确指出现有自知识蒸馏方法对标签噪声敏感、泛化能力差，"
        "并提出了门控集成自教师网络 + 基于影响函数的标签噪声去除方法。这表明 DTSKD 团队自身已认识到"
        "SKD 在噪声下的脆弱性问题。但 GEIKD 采用了与 DTSKD 完全不同的架构（门控 BiFPN 集成 vs 双教师预测矩阵），"
        "两者是平行的方法改进，而非对 DTSKD 机制的噪声行为分析。"
    )

    # ---- Section 2.3 ----
    add_heading_styled(doc, "2.3 噪声鲁棒自蒸馏与知识蒸馏方法", level=2)
    add_body_text(doc,
        "近年来涌现了大量针对标签噪声场景的知识蒸馏和自蒸馏方法，可归纳为以下几类："
    )
    add_body_text(doc,
        "（1）自蒸馏 + 噪声过滤：NoRD (Sharma et al., Applied Intelligence 2025) 提出相对自监督 + 决策匹配，"
        "在 >50% 噪声率下比 SOTA 鲁棒损失函数提升 8-10%；SNTD (Lan et al., TPAMI 2025) 提出 Self-Not-True "
        "Distillation——通过在 logits 中遮蔽真实类别，迫使网络从非真实类中获取纠错知识，并引入逐类蒸馏和动态权重调整。"
    )
    add_body_text(doc,
        "（2）知识蒸馏 + 标签校正：KDMLC (黄贻望等, 计算机研究与发展 2024) 将元标签校正模型作为教师，通过在线蒸馏"
        "指导轻量学生，高噪声水平下比 MLC 提升 5.50%；Ambiguity-Guided Mutual Label Refinery (Jiang et al., "
        "TNNLS 2023) 提出模糊引导的教师-学生互标签精炼；RMDNet (Chen et al., 2024) 通过自监督对比学习建模"
        "样本间关系，以知识蒸馏方式校准噪声样本的表征分布。"
    )
    add_body_text(doc,
        "（3）历史知识融合：LFB (Shi et al., Expert Systems with Applications 2025) 提出动态选择最佳历史权重配合"
        "历史 Logits 对比损失；Forget Me Not (Stern et al., TPAMI 2026) 利用训练历史中的多个 checkpoint 聚合为集成，"
        "再蒸馏回单一模型，在标签噪声场景下实现训练/推理复杂度降低与性能提升的「双赢」；"
        "DPSD (Li et al., IEEE Access 2024) 提出时维解耦与 Zipf 分布校准历史知识。"
    )

    # Comparison table
    add_styled_table(doc,
        ["方法", "年份", "发表", "核心思想", "与本项目关系"],
        [
            ["Das & Sanghavi", "2023", "ICML", "SD+噪声的理论分析（ξ-mixing, anti-learning）", "理论基础，数学形式相似"],
            ["GEIKD", "2023", "CVIU", "门控集成 + 影响函数去噪（DTSKD团队）", "最相关方法，不同架构"],
            ["Chung et al.", "2024", "arXiv", "SD = 标签平均，多轮收益递减", "理论框架"],
            ["Wu et al.", "2025", "arXiv", "自训练U型风险：去噪 vs 信号遗忘", "理论框架"],
            ["NoRD", "2025", "Applied Intelligence", "相对自监督 + 决策匹配", "噪声鲁棒SD方法"],
            ["SNTD", "2025", "TPAMI", "Self-Not-True + 逐类蒸馏", "噪声鲁棒SD方法"],
            ["KDMLC", "2024", "计算机研究与发展", "元标签校正 + 在线蒸馏", "中文文献，在线蒸馏"],
            ["LFB", "2025", "Expert Systems with Applications", "动态最佳历史权重 + HLC loss", "历史知识利用方法"],
            ["Forget Me Not", "2025", "TPAMI (2026)", "Checkpoint融合 + 蒸馏", "历史checkpoint利用"],
            ["DPSD", "2024", "IEEE Access", "时维解耦 + Zipf 校准", "历史知识校准"],
        ],
        col_widths=[2.5, 1.0, 2.5, 4.5, 3.5]
    )

    # ---- Section 2.4 ----
    add_heading_styled(doc, "2.4 诚实的研究空白评估", level=2)
    add_body_text(doc,
        "基于上述系统文献检索，本研究对创新性做出如下坦诚评估："
    )

    add_styled_table(doc,
        ["研究方向", "已有工作", "状态", "本研究差异化"],
        [
            ["SD + 标签噪声的理论分析", "Das (ICML 2023), Chung (2024), Wu (2025)", "✅ 已有扎实基础", "将理论洞察应用于 DTSKD 的具体机制分析"],
            ["噪声鲁棒 SKD 方法", "GEIKD (2023), NoRD (2025), SNTD (2025), KDMLC (2024)", "✅ 已有多种方法", "不做方法改进，做 DTSKD 的机理分析"],
            ["DTSKD 双教师在噪声下的消融分解", "无已有工作", "✅ 核心空白", "CE/HistKD/StructKD/Full 四路对比"],
            ["DTSKD 预测矩阵反馈循环的噪声行为", "无已有工作", "✅ 核心空白", "动态更新的 all_predictions 在噪声下的错误累积曲线"],
            ["SD 在噪声下退化 vs CE 的系统基准", "Das(2023)有理论对比，NoRD(2025)有实验对比", "⚠️ 部分已有", "聚焦 DTSKD 架构，更细粒度消融"],
        ],
        col_widths=[3.5, 3.5, 2.5, 4.5]
    )

    # ---- Section 2.5 ----
    add_heading_styled(doc, "2.5 本研究的确切定位", level=2)
    add_body_text(doc,
        "本研究不声称「自知识蒸馏在标签噪声下的行为无人研究」——已有 Das (ICML 2023)、"
        "Chung (2024)、Wu (2025) 等扎实的理论工作，以及 GEIKD (2023)、NoRD (2025)、"
        "SNTD (TPAMI 2025) 等多种方法。"
    )
    add_body_text(doc,
        "但以下三个问题的答案仍然是未知的："
    )
    add_body_text(doc,
        "第一，DTSKD 的双教师架构——历史教师（动态预测矩阵）与结构教师（backbone 到分支的蒸馏）——"
        "在噪声下各自的鲁棒性表现如何？现有理论分析的是标准单教师 SD，现有方法多提出新的噪声鲁棒架构，"
        "但没有工作对 DTSKD 的两教师做受控噪声下的消融分解。"
    )
    add_body_text(doc,
        "第二，DTSKD 特有的预测矩阵反馈循环——all_predictions 在每个 batch 被覆写，形成跨 epoch "
        "的自举式更新——在噪声下是否会产生比静态 SD 更严重的错误累积？"
        "Das (2023) 分析的是静态教师（训完固定），Wu (2025) 分析的是多轮重训，"
        "均未涉及这种训练过程中每 batch 更新的在线式反馈。"
    )
    add_body_text(doc,
        "第三，GEIKD (2023) 虽然来自 DTSKD 团队且针对噪声问题，但它是一种完全不同的架构（门控 BiFPN 集成），"
        "研究动机是提出新方法而非分析 DTSKD 的噪声行为。"
        "因此，一个对 DTSKD 的 History Teacher 和 Structure Teacher 在噪声下的系统基准测试，"
        "仍然是有价值的——它填补了「已知 SD 在噪声下的行为」与「DTSKD 这一具体架构的噪声特性」之间的鸿沟。"
    )
    add_body_text(doc,
        "简而言之：已有工作告诉我们「自蒸馏在噪声下会怎样」（理论）和「该如何改进」（方法），"
        "但尚未回答「DTSKD 的每个组件在噪声下贡献了什么」。"
        "本研究通过四路消融实验，提供这一分解性的实证答案。"
    )

    doc.add_page_break()

    # ============================================================
    # CHAPTER 3: 研究假设
    # ============================================================
    add_heading_styled(doc, "三、研究假设", level=1)

    add_heading_styled(doc, "3.1 主假设", level=2)
    add_body_text(doc,
        "H₁：DTSKD 在标签噪声下的性能退化速率 > 普通 CE 训练的性能退化速率。"
        "机制解释：标签噪声通过双通道污染 DTSKD——（1）直接通道：软目标中的 one-hot 分量包含错误标签，"
        "导致损失函数的监督信号错误；（2）反馈通道：被噪声标签训练的模型产生错误预测 → "
        "存储到历史预测矩阵 → 下一轮作为教师信号使用 → 错误循环放大。"
    )

    add_heading_styled(doc, "3.2 子假设", level=2)
    add_styled_table(doc,
        ["假设编号", "陈述", "验证方式"],
        [
            ["H₁ₐ", "HistKD Only 在噪声下退化最快（纯历史教师无补偿）", "HistKD vs CE 退化斜率对比"],
            ["H₁b", "StructKD Only 在噪声下与 CE 退化相似（仅依赖当前 backbone）", "StructKD vs CE 退化斜率对比"],
            ["H₁c", "Full DTSKD 中 StructKD 部分补偿 HistKD 的噪声放大", "Full DTSKD vs HistKD 退化斜率对比"],
        ],
        col_widths=[1.5, 6.0, 6.5]
    )

    add_heading_styled(doc, "3.3 竞争假设", level=2)
    add_body_text(doc,
        "H₂：DTSKD 在噪声下比 CE 更鲁棒（备择假设）。机制解释：EMA 式软目标 α_t·y + (1-α_t)·ŷ "
        "天然构成标签平滑（Label Smoothing），已知标签平滑对标签噪声有一定鲁棒性；"
        "多分支结构提供集成效应，不同分支对噪声的响应可能互补；"
        "结构教师（backbone 输出）基于当前输入而非历史，不受标签噪声直接影响。"
    )

    doc.add_page_break()

    # ============================================================
    # CHAPTER 4: 实验设计
    # ============================================================
    add_heading_styled(doc, "四、实验设计", level=1)

    add_heading_styled(doc, "4.1 实验框架", level=2)
    add_body_text(doc,
        "本实验严格遵循验证性研究范式：不修改模型架构，不设计新算法或损失函数，不添加新模块，"
        "仅注入噪声，观察各方法在噪声下的行为差异。这是本研究的核心方法论原则——首先理解问题，再考虑解决。"
    )

    add_heading_styled(doc, "4.2 实验因子", level=2)

    add_body_text(doc, "因子一：训练方法（4 个 Baselines）", first_line_indent=False)
    add_styled_table(doc,
        ["方法", "HSKD", "ce_weight", "kd_weight", "含义"],
        [
            ["CE Only", "0", "1.0", "0.0", "纯交叉熵，无任何蒸馏"],
            ["StructKD Only", "0", "0.2", "0.8", "仅结构蒸馏（分支学 backbone）"],
            ["HistKD Only", "1", "1.0", "0.0", "仅历史蒸馏（上一轮预测作软标签）"],
            ["Full DTSKD", "1", "0.2", "0.8", "双教师完整版（论文原始配置）"],
        ],
        col_widths=[3.0, 1.5, 2.0, 2.0, 5.5]
    )

    add_body_text(doc, "因子二：标签噪声（Phase 1 — 核心实验）", first_line_indent=False)
    add_body_text(doc,
        "采用对称标签噪声（Symmetric Label Noise），对训练集中 r ∈ {0%, 10%, 20%, 40%} 比例的样本，"
        "将其标签随机翻转为任意其他类别（均匀分布）。噪声在数据集创建时一次性生成，固定于每个样本"
        "（非每 epoch 随机重采样）。使用相同随机种子（seed=27）为所有 4 个 baseline 生成完全相同的噪声模式，"
        "确保严格公平对比。验证集始终使用干净标签。"
    )

    add_body_text(doc, "因子三：输入损坏（Phase 2 — 扩展实验）", first_line_indent=False)
    add_styled_table(doc,
        ["损坏类型", "强度梯度", "操作空间"],
        [
            ["高斯噪声", "σ ∈ {0.05, 0.1, 0.2}", "归一化后张量空间 [0, 1]"],
            ["高斯模糊", "kernel ∈ {3, 5}", "PIL 图像空间"],
            ["椒盐噪声", "p ∈ {0.05, 0.1}", "PIL 图像空间"],
        ],
        col_widths=[3.5, 4.5, 6.0]
    )

    add_heading_styled(doc, "4.3 固定控制变量", level=2)
    add_styled_table(doc,
        ["参数", "值", "控制目的"],
        [
            ["数据集", "CIFAR-100", "与 DTSKD 论文一致"],
            ["模型", "ResNet18-DTSKD", "标准配置"],
            ["训练轮数", "50 epochs", "快速筛选（与已完成消融实验一致）"],
            ["批大小", "64", "RTX 4050 6GB 显存限制"],
            ["学习率", "0.1（恒定，无衰减）", "隔离 lr 调度效应"],
            ["随机种子", "27", "可复现"],
            ["α_t 调度", "cos, cos_max=0.9, cos_min=0.0, α_end_epoch=300", "α_t 在 50 epoch 内维持 [0.84, 0.90]"],
            ["workers", "0", "单进程，避免 fork 导致噪声不一致"],
        ],
        col_widths=[3.5, 5.5, 5.0]
    )

    add_heading_styled(doc, "4.4 噪声注入技术方案", level=2)
    add_body_text(doc,
        "代码修改范围仅涉及 2 个文件：main.py 新增 4 个 CLI 参数（label_noise_rate, label_noise_seed, "
        "input_corruption, corruption_level），loader/custom_dataloader.py 新增噪声注入逻辑。"
        "标签噪声注入位于 custom_dataloader.py 的 CIFAR-100 数据集创建后、DataLoader 创建前，"
        "通过修改 trainset.targets 列表实现。输入损坏注入位于 transform_train 的 Compose 流水线中，"
        "条件性地插入损坏变换（PIL 空间：模糊和椒盐噪声在 ToTensor 前；Tensor 空间：高斯噪声在 "
        "ToTensor 后、Normalize 前）。验证集始终保持干净。"
    )

    add_heading_styled(doc, "4.5 评估指标", level=2)
    add_styled_table(doc,
        ["指标", "定义", "用途"],
        [
            ["Val Top-1 Accuracy", "backbone 在干净验证集上的 top-1", "主要指标"],
            ["Val Top-5 Accuracy", "backbone 在干净验证集上的 top-5", "辅助指标"],
            ["Branch Accuracies", "b1/b2/b3 在验证集上的 top-1", "分支噪声敏感性差异"],
            ["退化斜率 Δ(r)", "Acc(0%) − Acc(r%)", "核心对比指标"],
            ["相对退化 Δ_rel(r)", "Δ(r) / Acc(0%)", "归一化后跨方法对比"],
            ["过拟合差距", "Train Acc − Val Acc", "对噪声标签的记忆化程度"],
            ["训练曲线", "Val Acc vs Epoch", "退化是否随训练累积"],
        ],
        col_widths=[3.5, 5.5, 5.0]
    )

    add_heading_styled(doc, "4.6 实验矩阵", level=2)

    add_body_text(doc, "Phase 1：标签噪声（16 个实验，预计约 16 小时）", first_line_indent=False)
    add_body_text(doc, "4 种方法 × 4 种噪声率（0%, 10%, 20%, 40%）= 16 个实验。这是验证核心假设的最小必要集。")

    # Phase 1 table
    add_styled_table(doc,
        ["实验名", "HSKD", "ce/kd_weight", "label_noise_rate"],
        [
            ["noise_label_00_ce", "0", "1.0 / 0.0", "0%"],
            ["noise_label_10_ce", "0", "1.0 / 0.0", "10%"],
            ["noise_label_20_ce", "0", "1.0 / 0.0", "20%"],
            ["noise_label_40_ce", "0", "1.0 / 0.0", "40%"],
            ["noise_label_00_struct", "0", "0.2 / 0.8", "0%"],
            ["noise_label_10_struct", "0", "0.2 / 0.8", "10%"],
            ["noise_label_20_struct", "0", "0.2 / 0.8", "20%"],
            ["noise_label_40_struct", "0", "0.2 / 0.8", "40%"],
            ["noise_label_00_hist", "1", "1.0 / 0.0", "0%"],
            ["noise_label_10_hist", "1", "1.0 / 0.0", "10%"],
            ["noise_label_20_hist", "1", "1.0 / 0.0", "20%"],
            ["noise_label_40_hist", "1", "1.0 / 0.0", "40%"],
            ["noise_label_00_full", "1", "0.2 / 0.8", "0%"],
            ["noise_label_10_full", "1", "0.2 / 0.8", "10%"],
            ["noise_label_20_full", "1", "0.2 / 0.8", "20%"],
            ["noise_label_40_full", "1", "0.2 / 0.8", "40%"],
        ],
        col_widths=[4.0, 1.5, 2.5, 2.5]
    )

    add_body_text(doc, "Phase 2：输入损坏（14 个实验，预计约 12 小时）", first_line_indent=False)
    add_body_text(doc, "仅跑 CE 和 Full DTSKD（最大对比度），覆盖高斯噪声（σ=0.05, 0.1, 0.2）、"
                 "高斯模糊（kernel=3, 5）、椒盐噪声（p=0.05, 0.1）。")

    add_body_text(doc, "Phase 3：统计验证（6-8 个实验，预计约 6 小时）", first_line_indent=False)
    add_body_text(doc, "对关键发现用额外 3 个随机种子（42, 123, 2023）复现，排除单种子偶然性。")

    add_heading_styled(doc, "4.7 统计考虑", level=2)
    add_body_text(doc,
        "本研究坦诚声明以下统计局限：（1）每个条件仅 1 次运行（seed=27），无法计算置信区间——"
        "这是快速筛选阶段的有意权衡，Phase 3 将用多种子修复；（2）16 组对比同时进行，假阳性风险增加——"
        "本研究的目标是发现效应方向（定性），而非精确估计效应大小（定量）；（3）仅在 CIFAR-100 上实验——"
        "若发现显著效应，需在 CIFAR-10、Tiny-ImageNet 上验证泛化性；（4）50 epoch 限制——"
        "短训练可能不足以让反馈循环效应完全展现，若未观察到差异需扩展至 300 epoch。"
    )

    doc.add_page_break()

    # ============================================================
    # CHAPTER 5: 预期结果与解释框架
    # ============================================================
    add_heading_styled(doc, "五、预期结果与解释框架", level=1)

    add_heading_styled(doc, "5.1 结果-解释映射表", level=2)
    add_body_text(doc,
        "本研究最重要的贡献之一是提供了一个完整的分析框架。无论观察结果如何，都有明确的理论解释和学术贡献路径。"
    )

    add_styled_table(doc,
        ["观察到的现象", "理论解释", "学术贡献"],
        [
            ["HistKD 退化 > CE 退化", "历史教师传播错误，正反馈循环放大", "首次证明历史机制脆弱性"],
            ["HistKD 退化 ≈ CE 退化", "EMA 标签平滑效应抵消错误传播", "排除'一定更脆弱'的担忧"],
            ["HistKD 退化 < CE 退化", "时序平滑提供隐式正则化", "为 DTSKD 提供优势论证"],
            ["Full DTSKD 退化 < HistKD 退化", "结构教师补偿历史教师噪声敏感性", "双教师互补性发现"],
            ["StructKD 退化 ≈ CE 退化", "结构蒸馏不引入额外噪声敏感性", "结构教师是'安全的'"],
            ["退化随 epoch 加速", "正反馈循环确实存在", "动态 α_t 或质量门控的必要性"],
            ["Branch 退化不一致", "浅层对输入更敏感，深层对标签更敏感", "分支级自适应策略设计依据"],
        ],
        col_widths=[4.0, 4.5, 4.5]
    )

    add_heading_styled(doc, "5.2 论文潜力评估", level=2)
    add_styled_table(doc,
        ["结果模式", "发表潜力", "目标期刊"],
        [
            ["HistKD 显著退化 + Full DTSKD 有补偿", "高", "PR, TNNLS, NeurIPS"],
            ["DTSKD 在噪声下更鲁棒（意外发现）", "高", "同上，作为鲁棒性论证"],
            ["无显著差异（接近零发现）", "低", "作为技术报告/补充材料"],
            ["HistKD 退化但 CE 也退化（无区分度）", "中", "需更多分析角度"],
        ],
        col_widths=[5.5, 2.5, 5.0]
    )

    doc.add_page_break()

    # ============================================================
    # CHAPTER 6: 实施计划
    # ============================================================
    add_heading_styled(doc, "六、实施计划", level=1)

    add_body_text(doc,
        "项目分阶段推进，每阶段完成后评估结果再决定是否进入下一阶段，降低一次性投入风险。"
    )

    add_styled_table(doc,
        ["阶段", "任务", "预计耗时", "产出"],
        [
            ["Phase 0", "实现噪声注入代码 + 验证", "2 hours", "可工作的噪声注入框架"],
            ["Phase 1", "标签噪声 16 实验", "16 hours", "核心结论（支持/否定假设）"],
            ["分析 1", "Phase 1 数据分析 + 决策", "3 hours", "是否继续 Phase 2 的决策"],
            ["Phase 2", "输入损坏 12 实验", "12 hours", "噪声类型差异分析"],
            ["分析 2", "综合分析 + 可视化", "4 hours", "图表 + 初步结论"],
            ["Phase 3", "多种子验证（如需要）", "6 hours", "统计显著性"],
            ["合  计", "", "≈ 37-43 hours", ""],
        ],
        col_widths=[2.0, 5.5, 3.0, 4.5]
    )

    add_body_text(doc, "代码变更清单：", first_line_indent=False)
    add_styled_table(doc,
        ["文件", "变更类型", "变更量"],
        [
            ["main.py", "新增 4 个 CLI 参数", "≈ 10 行"],
            ["loader/custom_dataloader.py", "新增噪声注入逻辑 + helper 函数", "≈ 60 行"],
            ["scripts/run_phase1.ps1", "批量运行脚本", "≈ 50 行"],
        ],
        col_widths=[5.0, 5.5, 2.5]
    )

    doc.add_page_break()

    # ============================================================
    # CHAPTER 7: 贡献陈述
    # ============================================================
    add_heading_styled(doc, "七、贡献陈述", level=1)

    add_heading_styled(doc, "若假设成立（DTSKD 噪声下退化更快）", level=2)
    points_true = [
        "首次揭示自知识蒸馏中历史教师机制的噪声脆弱性",
        "为设计鲁棒自知识蒸馏方法提供动机和基线",
        "建立非理想条件下评估 SKD 方法的实验范式",
    ]
    for i, pt in enumerate(points_true, 1):
        add_body_text(doc, f"（{i}）{pt}。")

    add_heading_styled(doc, "若假设不成立（DTSKD 更鲁棒）", level=2)
    points_false = [
        "为 DTSKD 提供鲁棒性维度的新优势论证",
        "揭示 EMA 软目标在噪声下的隐式正则化效应",
        "扩展 DTSKD 的适用场景（标注质量受限的任务）",
    ]
    for i, pt in enumerate(points_false, 1):
        add_body_text(doc, f"（{i}）{pt}。")

    add_heading_styled(doc, "无论结果如何", level=2)
    points_always = [
        "填补自知识蒸馏鲁棒性分析的研究空白",
        "提供系统的噪声注入实验框架，可复用于其他 SKD 方法",
        "明确 DTSKD 各组件（HistKD / StructKD）在噪声下的独立行为",
        "为后续方法设计提供分析依据",
    ]
    for i, pt in enumerate(points_always, 1):
        add_body_text(doc, f"（{i}）{pt}。")

    doc.add_page_break()

    # ============================================================
    # REFERENCES
    # ============================================================
    add_heading_styled(doc, "参考文献", level=1)

    references = [
        "[1] Das R, Sanghavi S. Understanding Self-Distillation in the Presence of Label Noise[C]. "
        "International Conference on Machine Learning (ICML), PMLR 202:7102-7140, 2023. "
        "[最直接相关的理论工作：分析了标准自蒸馏 (ξ-mixing) 在标签噪声下的 bias-variance tradeoff，"
        "发现高噪声下最优 ξ 可大于 1（anti-learning）。]",

        "[2] Liu F, Wang Y, Li Z, Pan Z. GEIKD: Self-Knowledge Distillation Based on Gated Ensemble "
        "Networks and Influences-Based Label Noise Removal[J]. Computer Vision and Image Understanding "
        "(CVIU), 2023, 235: 103771. [DTSKD 团队自身提出的噪声鲁棒 SKD 方法，使用门控 BiFPN 集成 + "
        "影响函数去噪。与 DTSKD 是并行架构，不是同一机制的分析。]",

        "[3] Chung S, et al. Rethinking Self-Distillation: Label Averaging and Enhanced Soft Label "
        "Refinement with Partial Labels[J]. arXiv:2402.10482, 2024/2025. "
        "[揭示自蒸馏等效于特征空间的标签平均，多轮 SD 收益递减，单轮 PLL 在高噪声下更优。]",

        "[4] Wu M, Yang A Y, Sun Q. Why Self-Training Helps and Hurts: Denoising vs. Signal "
        "Forgetting[J]. arXiv:2602.14029, 2025. "
        "[自训练的 U 型风险曲线：随机误差衰减 vs 系统性信号遗忘，提出迭代 GCV 早停准则。]",

        "[5] Sharma S, Lodhi S S, Srivastava V, Chandra J. NoRD: A Framework for Noise-Resilient "
        "Self-Distillation through Relative Supervision[J]. Applied Intelligence, 2025, 55: 452. "
        "[相对自监督 + 决策匹配框架，>50% 噪声率下比 SOTA 提升 8-10%。]",

        "[6] Lan L, Wang J, Wu X, Han B, Liu X. Continuous Review and Timely Correction: Enhancing "
        "the Resistance to Noisy Labels via Self-Not-True and Class-Wise Distillation[J]. "
        "IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2025/2026. "
        "[Self-Not-True Distillation：遮蔽真实类 logits 以获取纠错知识，引入逐类教师和动态权重。]",

        "[7] 黄贻望, 黄雨鑫, 刘声. 一种基于在线蒸馏的轻量化噪声标签学习方法 (KDMLC)[J]. "
        "计算机研究与发展, 2024, 61(12): 3121-3133. "
        "[中文文献：元标签校正模型作为教师，在线蒸馏指导轻量学生，高噪声下比 MLC 提升 5.50%。]",

        "[8] 黄雨鑫, 黄贻望, 黄辉. 基于浅层网络预测的元标签校正方法[J]. "
        "计算机应用, 2024, 44(11): 3364-3370. "
        "[中文文献：深层与浅层网络协同预测伪标签，知识蒸馏指导去噪。]",

        "[9] Li Z, Li X, Yang L, Song R, Yang J, Pan Z. Dual Teachers for Self-Knowledge "
        "Distillation[J]. Pattern Recognition, 2024, 151: 110422. "
        "[DTSKD 原始论文：双教师（历史+结构）自知识蒸馏。本文的被研究对象。]",

        "[10] Kim K, Ji B, Yoon D, Hwang S. Self-Knowledge Distillation with Progressive "
        "Refinement of Targets[C]. IEEE/CVF International Conference on Computer Vision (ICCV), 2021. "
        "[PS-KD：渐进式自知识蒸馏先驱，上一 epoch 预测作为软目标。]",

        "[11] Wang X, et al. Learn From the Past: Experience Ensemble Knowledge Distillation[J]. "
        "arXiv:2202.12488, 2022. "
        "[EEKD：利用教师训练过程中的多个历史 checkpoint 进行集成蒸馏。]",

        "[12] Cui Z, Pei H. How Is Uncertainty Propagated in Knowledge Distillation?[J]. "
        "arXiv:2601.18909, 2025. "
        "[蒸馏中不确定度传播的理论框架：inter-student vs intra-student 不确定度，多教师平均降噪。]",

        "[13] Shi J, et al. Learn from the Best: A Universal Self-Distillation Approach with "
        "Historical Logits[J]. Expert Systems with Applications, 2025. "
        "[LFB：动态选择最佳历史权重 + 历史 Logits 对比损失，具有抗噪能力。]",

        "[14] Stern O, Corn B, Weinshall D. Forget Me Not: Fighting Local Overfitting with "
        "Knowledge Fusion and Distillation[J]. IEEE Transactions on Pattern Analysis and Machine "
        "Intelligence (TPAMI), 2026 (to appear). "
        "[Checkpoint 融合 + 蒸馏对抗局部过拟合，标签噪声场景下效果显著。]",

        "[15] Chen X, et al. Relation Modeling and Distillation for Learning with Noisy Labels "
        "(RMDNet)[J]. arXiv:2405.19606, 2024. "
        "[自监督对比学习建模样本关系，知识蒸馏校准噪声样本表征。]",

        "[16] Jiang Z, et al. Knowledge Distillation Meets Label Noise Learning: Ambiguity-Guided "
        "Mutual Label Refinery[J]. IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2023. "
        "[模糊引导的教师-学生互标签精炼方法。]",

        "[17] Huang W, Ye M, Shi Z, Li H, Du B. Self-Knowledge Distillation with Dimensional "
        "History Knowledge[J]. Science China Information Sciences, 2025. "
        "[DimSelfKD：从维度视角蒸馏历史知识（intra-class + inter-class）。]",

        "[18] Li Z, et al. Decoupled Time-Dimensional Progressive Self-Distillation with Knowledge "
        "Calibration (DPSD)[J]. IEEE Access, 2024. "
        "[时维解耦 + Zipf 分布校准历史知识，解决历史输出低置信度问题。]",

        "[19] Kaplun G, et al. Knowledge Distillation: Bad Models Can Be Good Role Models[C]. "
        "Advances in Neural Information Processing Systems (NeurIPS), 2022. "
        "[噪声教师仍可产生优于标签的学生——conditional sampler 视角。]",

        "[20] Shen Y, Xu L, Yang Y, et al. Self-Distillation from the Last Mini-Batch for "
        "Consistency Regularization (DLB)[C]. IEEE/CVF Conference on Computer Vision and Pattern "
        "Recognition (CVPR), 2022. "
        "[上一 mini-batch 蒸馏用于一致性正则化，对标签噪声具有鲁棒性。]",
    ]

    for ref in references:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.first_line_indent = Cm(-0.74)
        p.paragraph_format.left_indent = Cm(0.74)
        run = p.add_run(ref)
        run.font.name = 'Times New Roman'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋')
        run.font.size = Pt(10)

    doc.add_page_break()

    # ============================================================
    # APPENDIX A
    # ============================================================
    add_heading_styled(doc, "附录 A：已完成的前期工作", level=1)
    add_styled_table(doc,
        ["工作", "状态", "关键发现"],
        [
            ["DTSKD 完整复现（300 epoch）", "✅ 完成", "val_top1=79.4%，超越论文报告的 74-76%"],
            ["HSKD 单 GPU Bug 修复", "✅ 完成", "单 GPU 下 all_predictions 更新缺失"],
            ["4 方法消融实验（50 epoch）", "✅ 完成", "HistKD (56.45%) > StructKD (55.27%) > CE (55.04%)"],
            ["α_t 调度分析", "✅ 完成", "余弦 α_t→0 导致教师崩塌，动态 α_t 防止崩塌"],
        ],
        col_widths=[4.5, 2.0, 7.5]
    )

    # ============================================================
    # APPENDIX B
    # ============================================================
    add_heading_styled(doc, "附录 B：实验命名规范", level=1)
    add_body_text(doc, "实验命名格式：noise_{type}_{level}_{method}", first_line_indent=False)
    add_styled_table(doc,
        ["字段", "含义", "取值"],
        [
            ["type", "噪声类型", "label | gauss | blur | sp"],
            ["level", "噪声强度", "00 | 10 | 20 | 40 (label); 005 | 010 | 020 (gauss); 3 | 5 (blur); 005 | 010 (sp)"],
            ["method", "训练方法", "ce | struct | hist | full"],
        ],
        col_widths=[2.0, 3.0, 9.0]
    )
    add_body_text(doc, "示例：noise_label_20_full = 20% 标签噪声下的 Full DTSKD；noise_gauss_010_ce = σ=0.1 高斯噪声下的 CE Only。", first_line_indent=False)

    # ============================================================
    # SAVE
    # ============================================================
    output_path = r"D:\Python_code\DTSKD\DTSKD噪声鲁棒性研究_开题报告.docx"
    doc.save(output_path)
    print(f"开题报告已保存至：{output_path}")


if __name__ == "__main__":
    create_report()
