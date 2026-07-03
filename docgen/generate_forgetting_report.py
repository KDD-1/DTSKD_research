"""
生成「SKD 遗忘问题」方向的开题报�?Word 文档
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import datetime


def set_cell_shading(cell, color):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def add_heading_styled(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = '黑体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        if level == 1: run.font.size = Pt(16)
        elif level == 2: run.font.size = Pt(14)
        elif level == 3: run.font.size = Pt(13)
    return h


def add_body(doc, text, indent=True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_after = Pt(4)
    pf.space_before = Pt(2)
    if indent:
        pf.first_line_indent = Cm(0.74)
    run = p.add_run(text)
    run.font.name = '仿宋'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋')
    run.font.size = Pt(12)
    return p


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1+len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]; cell.text = ''
        r = cell.paragraphs[0].add_run(h)
        r.font.name = '黑体'; r._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        r.font.size = Pt(9); r.bold = True; r.font.color.rgb = RGBColor(255,255,255)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_shading(cell, "1F4E79")
    for ri, row in enumerate(rows):
        bg = "F2F7FB" if ri % 2 == 0 else "FFFFFF"
        for ci, txt in enumerate(row):
            cell = table.rows[ri+1].cells[ci]; cell.text = ''
            r = cell.paragraphs[0].add_run(str(txt))
            r.font.name = '仿宋'; r._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋')
            r.font.size = Pt(9)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_cell_shading(cell, bg)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows: row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return table


def create_report():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0); sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.54); sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(3.17); sec.right_margin = Cm(3.17)

    style = doc.styles['Normal']
    style.font.name = '仿宋'; style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋')

    # ============ COVER ============
    for _ in range(6): doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("开 �?�?�?); r.font.name = '黑体'
    r._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    r.font.size = Pt(26); r.bold = True

    doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("自知识蒸馏中的遗忘问题：持续学习视角下的分析与对�?)
    r.font.name = '黑体'; r._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    r.font.size = Pt(18); r.bold = True

    doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("�?基于 Wu (ICML 2026) 理论�?Stern (AAAI 2025) 度量的实证研�?�?)
    r.font.name = 'Times New Roman'; r.font.size = Pt(12); r.italic = True

    for _ in range(4): doc.add_paragraph()
    for label, val in [
        ("研究方向", "机器学习 · 自知识蒸�?· 遗忘问题 · 持续学习"),
        ("理论支撑", "Wu et al. (ICML 2026) + Stern et al. (AAAI 2025)"),
        ("日　　�?, datetime.date.today().strftime("%Y�?m�?d�?)),
    ]:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(f"{label}：{val}")
        r.font.name = '仿宋'; r._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋')
        r.font.size = Pt(14)
    doc.add_page_break()

    # ============ ABSTRACT ============
    add_heading_styled(doc, "�? �?, 1)
    add_body(doc,
        "自知识蒸馏（Self-Knowledge Distillation, SKD）通过模型自身的历史预测作为软监督信号�?
        "在图像分类等任务上取得了显著成功。然而，最新理论工�?(Wu et al., ICML 2026) 揭示�?
        "迭代自训练中的一个根本性权衡：随机噪声随迭代被逐步消除（去噪效应），但真实信号也会"
        "在弱特征方向上被逐渐丢弃（信号遗忘效应）。两者竞争产�?U 型风险曲线—�?
        "即过度自蒸馏会导致性能退化。与此同时，Stern et al. (AAAI 2025) 提出�?
        "「局部过拟合」和 Forget Fraction 度量，为检测训练过程中的样本级遗忘提供了操作化工具�?
    )
    add_body(doc,
        "本研究从持续学习（Continual Learning, CL）的视角审视 SKD 中的遗忘问题�?
        "持续学习领域为解决「灾难性遗忘」已发展出多种成熟技术——EWC、MAS、Experience Replay 等�?
        "本研究提出：这些抗遗忘技术可以跨越领域边界，被重新适配用于修复自知识蒸馏内部的信号遗忘�?
        "关键创新在于引入「教师置信度加权」约束——在历史教师确信的样本上施加更强的抗遗忘约束�?
        "在教师不确定的样本上放松约束以允许新知识的学习�?
    )
    add_body(doc,
        "本研究的第一阶段为实证分析——在 CIFAR-100 上对 CE、HistKD �?Full DTSKD 三种训练范式�?
        "使用 Forget Fraction 度量系统刻画 SKD 的遗忘动力学，检验遗忘是否因历史教师的反馈循�?
        "而加剧。若实证阶段确认 SKD 特有的遗忘模式，第二阶段将适配 CL 抗遗忘技术进行修复�?
    )
    add_body(doc, "关键词：自知识蒸馏；遗忘问题；持续学习；信号遗忘；Forget Fraction；教师置信度")
    doc.add_page_break()

    # ============ CHAPTER 1 ============
    add_heading_styled(doc, "一、研究背景与问题提出", 1)

    add_heading_styled(doc, "1.1 自知识蒸馏的成功与隐�?, 2)
    add_body(doc,
        "自知识蒸馏（SKD）消除了传统知识蒸馏对外部预训练教师的依赖。其核心机制是："
        "模型在当前训练轮次产生的预测被存储，作为下一轮训练的软目标（软标签）�?
        "PS-KD (ICCV 2021)、CS-KD (AAAI 2021)、DTSKD (PR 2024) 等方法在 CIFAR-100 �?ImageNet "
        "上均取得了超越基线交叉熵训练的性能�?
    )
    add_body(doc,
        "然而，这种「自举式」（bootstrapping）训练范式存在一个深层的结构性问题："
        "历史预测作为软目标的质量决定�?SKD 的有效性。当模型在训练后期对某些样本的预测质�?
        "下降时，这些低质量的预测会被反馈到下一轮训练中——形成「错误循环」�?
        "Wu et al. (ICML 2026) 从理论上严格证明了这一现象——他们称之为「信号遗忘�?
        "（Signal Forgetting）——并指出这是迭代自训练的一个根本性局限，而非工程问题�?
    )

    add_heading_styled(doc, "1.2 理论基础：信号遗忘与 U 型风�?, 2)
    add_body(doc,
        "Wu et al. (ICML 2026) 在过参数化线性回归框架下分析了迭代自训练的动力学�?
        "核心发现是预测风险可分解为两个竞争分量：系统性偏�?B(t)²（信号遗忘，随迭代递增�?
        "和随机方�?V(t)（去噪，随迭代递减）。两者的竞争产生 U 型风险曲线—�?
        "存在一个最优的「自训练深度」，超过该深度后继续自蒸馏反而有害�?
    )
    add_body(doc,
        "迭代自训练等效于对特征方向施加迭代相关的谱滤波：强特征方向的信号被保留，"
        "弱特征方向的信号随迭代指数衰减。这�?Ridge 回归有本质区别——后者对所有方�?
        "均匀收缩，而自训练是自适应的：强信号保留更多，弱信号丢弃更多�?
    )
    add_body(doc,
        "将这一框架映射�?SKD：PS-KD/DTSKD 中的软目标公�?soft_target = α_t·y + (1-α_t)·ŷ_prev "
        "�?α_t 较大时（训练早期）接近纯监督学习，信号遗忘不显著；但�?α_t 衰减到较低水平后"
        "（训练后期），SKD 趋近于纯自训练，信号遗忘开始加速。这提出了一个关键问题："
        "在真实深度网络训练中，这种理论预测的遗忘效应是否确实存在？如果存在，"
        "它与 α_t 的衰减有何关系？SKD 的遗忘模式是否与普�?CE 训练不同�?
    )

    add_heading_styled(doc, "1.3 从持续学习看 SKD 的遗�?, 2)
    add_body(doc,
        "持续学习（Continual Learning）领域的核心问题是「灾难性遗忘」—�?
        "模型在学习新任务时会遗忘旧任务的知识。该领域已发展出多种成熟的抗遗忘技术："
        "弹性权重巩固（EWC, PNAS 2017）基�?Fisher 信息矩阵约束重要参数的更新；"
        "突触智能累积（MAS, ECCV 2018）使用参数敏感度衡量重要性；"
        "经验回放（Experience Replay）通过存储和重放旧样本来维持旧知识�?
        "梯度情景记忆（GEM, NeurIPS 2017）约束梯度方向不与旧任务冲突�?
    )
    add_body(doc,
        "SKD 中的遗忘�?CL 中的遗忘在本质上是相似的——都是模型在训练过程�?
        "丢失了之前已获得的正确知识。关键区别在于：CL 的遗忘发生在跨任务之间，"
        "�?SKD 的遗忘发生在单任务训练内部。这使得 SKD 的遗忘更加隐蔽—�?
        "它在全局验证精度可能仍在提升时就已经在局部发生了（即 Stern 等人所谓的「局部过拟合」）�?
    )
    add_body(doc,
        "本研究提出一个跨领域的核心假设：持续学习中的抗遗忘技术可以被重新适配到自知识蒸馏中，"
        "用于修复训练内部的信号遗忘问题。这个方向在 CL �?SKD 两个社区均未被探索�?
    )

    add_heading_styled(doc, "1.4 核心研究问题", 2)

    add_table(doc,
        ["编号", "研究问题", "验证方式"],
        [
            ["RQ1", "SKD 训练中是否存在显著的样本级遗忘？是否�?CE 训练更严重？",
             "使用 Stern �?Forget Fraction 度量�?CIFAR-100 上对 CE/HistKD/Full DTSKD 做对�?],
            ["RQ2", "SKD 的遗忘是否与 α_t 衰减相关？是否存�?α_t 安全阈值？",
             "分析 F_e 曲线�?α_t 衰减的时间对齐关�?],
            ["RQ3", "被遗忘的样本有什么特征？是否与历史教师的低置信度相关�?,
             "对被遗忘样本做特征分析：类别分布、难度、教师置信度/�?],
            ["RQ4", "CL 抗遗忘技术（EWC/MAS）能否适配�?SKD 中以缓解遗忘�?,
             "Phase 2：设计教师置信度加权�?EWC/MAS 变体并实验验�?],
        ],
        col_widths=[1.0, 6.0, 7.0]
    )
    doc.add_page_break()

    # ============ CHAPTER 2 ============
    add_heading_styled(doc, "二、文献综�?, 1)

    add_heading_styled(doc, "2.1 自训练的遗忘理论", 2)
    add_body(doc,
        "Wu et al. (ICML 2026) 提供了第一个严格的迭代自训练遗忘理论。他们在过参数化"
        "线性回归框架下证明了预测风险可分解为系统性偏差（信号遗忘）和随机方差（去噪）�?
        "两者竞争产�?U 型风险曲线。迭代自训练等效于迭代相关的谱滤波器——强特征方向保留�?
        "弱特征方向指数衰减。该文还提出了迭代广义交叉验证（GCV）准则用于数据驱动的早停�?
    )
    add_body(doc,
        "Stern et al. (AAAI 2025 / TPAMI 2026) 从实验角度独立发现了相关现象。他们提�?
        "「局部过拟合」概念——全局测试精度上升时，某些数据子区域的精度在下降—�?
        "并定义了 Forget Fraction 度量（F_e）来量化这一效应。他们的实验覆盖�?ConvNet �?ViT "
        "�?CIFAR-100、TinyImageNet、ImageNet 上的行为，发现模型容量越大，遗忘越严重�?
        "此外，他们还提出�?Knowledge Fusion + Distillation 的两阶段修复方法�?
    )

    add_heading_styled(doc, "2.2 持续学习中的抗遗忘技�?, 2)
    add_table(doc,
        ["方法", "发表", "核心机制", "�?SKD 中的潜在适配"],
        [
            ["EWC", "PNAS 2017", "Fisher 信息矩阵约束重要参数更新", "用教师置信度替代 Fisher 信息衡量参数重要�?],
            ["MAS", "ECCV 2018", "参数敏感度累积衡量重要�?, "用历史预测对参数的梯度衡量敏感度"],
            ["LwF", "TPAMI 2017", "旧模型输出作为软目标约束新模�?, "SKD 本身已经在做 LwF！但旧模型在遗忘"],
            ["Experience Replay", "ICML 2017", "存储和重放旧样本", "不需存样本——存历史教师的预测分布即�?],
            ["GEM", "NeurIPS 2017", "约束梯度方向不与旧任务冲�?, "约束梯度不与历史预测方向冲突"],
        ],
        col_widths=[3.0, 2.0, 4.5, 5.0]
    )

    add_heading_styled(doc, "2.3 自知识蒸馏与遗忘的交叉地�?, 2)
    add_body(doc,
        "MOSE (Yan et al., CVPR 2024) 提出�?Reverse Self-Distillation——在持续学习中使�?
        "自蒸馏来防止跨任务遗忘。这表明 CL 社区已经意识到自蒸馏可以作为抗遗忘工具�?
        "但反向的视角——用 CL 技术来修复自蒸馏内部的遗忘——尚无研究�?
    )
    add_body(doc,
        "Das & Sanghavi (ICML 2023) �?bias-variance tradeoff 角度分析了自蒸馏在标签噪声下的行为，"
        "暗示了在噪声条件下自蒸馏可能不如预期。但这与信号遗忘是不同的机制—�?
        "前者是标签噪声引入的偏差，后者是自蒸馏迭代中真实信号的丢失�?
    )

    add_heading_styled(doc, "2.4 研究空白与创新性定�?, 2)
    add_table(doc,
        ["方向", "已有工作", "状�?, "本研究差异化"],
        [
            ["迭代自训练的遗忘理论", "Wu (ICML 2026)", "�?已有理论", "将理论映射到真实 SKD 系统，实验验�?],
            ["局部过拟合�?Forget Fraction", "Stern (AAAI 2025)", "�?已有度量", "将度量应用于 SKD vs CE 的对比分�?],
            ["SKD 中遗忘的实验分析", "无已有工�?, "�?核心空白", "首次系统刻画 SKD 训练的遗忘动力学"],
            ["CL 技术适配�?SKD", "无已有工�?, "�?核心空白", "跨领域迁移：EWC/MAS �?SKD 内部遗忘修复"],
            ["教师置信度加权的遗忘约束", "无已有工�?, "�?方法创新", "核心技术创新点"],
        ],
        col_widths=[3.5, 3.0, 2.0, 5.5]
    )
    doc.add_page_break()

    # ============ CHAPTER 3 ============
    add_heading_styled(doc, "三、研究方�?, 1)

    add_heading_styled(doc, "3.1 Phase 1：遗忘检测实�?, 2)
    add_body(doc,
        "实验配置：CIFAR-100 数据集，ResNet18 架构�?00 epoch 完整训练�?
        "对比 CE Only、HistKD Only、Full DTSKD 三种训练范式，每�?3 个随机种子（27, 42, 123），"
        "�?9 个实验。每�?epoch 结束后在验证集上记录每个样本的分类正确性（per-sample correctness），"
        "存储为布尔张量（300 epochs × 10000 samples × 1 byte = 3MB）�?
    )
    add_body(doc,
        "核心分析指标：Forget Fraction F_e（在 epoch e 正确但最终错误的样本比例）�?
        "Learn Fraction L_e（在 epoch e 错误但最终正确的样本比例）�?
        "遗忘速率 dF/de（F_e 的离散导数）�?
        "F_e �?α_t �?Pearson 相关性�?
        "被遗忘样本的特征分析：类别分布、难度（早期置信度）、教师预测熵�?
    )

    add_heading_styled(doc, "3.2 Phase 2：CL 技术适配（如 Phase 1 确认 SKD 存在显著遗忘�?, 2)
    add_body(doc,
        "核心创新：教师置信度加权 EWC/MAS。标�?EWC 对所有样本的参数约束强度相同�?
        "本研究提出：在教师置信度高的样本上施加更强的约束（保护模型不遗忘已掌握的知识），"
        "在教师置信度低的样本上放松约束（允许模型更新其预测）�?
    )
    add_body(doc,
        "具体实现：对于每�?batch，计算历史教师对各样本的预测置信�?c_i = max(ŷ_prev[i])�?
        "构造加�?Fisher 信息矩阵 F_weighted = Σ_i c_i · F_i，其�?F_i 是样�?i �?Fisher�?
        "EWC 损失变为 L_ewc = Σ_j F_weighted[jj] · (θ_j - θ*_j)²�?
        "高置信度样本的参数变化被更严格地惩罚，低置信度样本的参数可以自由适应新知识�?
    )

    add_heading_styled(doc, "3.3 实施时间�?, 2)
    add_table(doc,
        ["阶段", "任务", "实验�?, "预计耗时", "产出"],
        [
            ["Phase 0", "代码实现（per-sample tracking + 分析脚本�?, "�?, "2h", "可运行的遗忘检测框�?],
            ["Phase 1a", "CE Only × 3 seeds × 300 epoch", "3", "~13.5h", "CE 基线遗忘曲线"],
            ["Phase 1b", "HistKD Only × 3 seeds × 300 epoch", "3", "~13.5h", "HistKD 遗忘曲线"],
            ["Phase 1c", "Full DTSKD × 3 seeds × 300 epoch", "3", "~13.5h", "Full DTSKD 遗忘曲线"],
            ["分析", "Phase 1 数据分析 + 可视�?, "�?, "5h", "遗忘动力学图�?+ 报告"],
            ["决策�?, "SKD 遗忘是否显著 > CE�?, "�?, "�?, "决定是否进入 Phase 2"],
            ["Phase 2", "CL 技术适配 + 实验验证", "TBD", "TBD", "方法论文"],
        ],
        col_widths=[2.0, 5.5, 1.5, 2.0, 4.0]
    )
    doc.add_page_break()

    # ============ CHAPTER 4 ============
    add_heading_styled(doc, "四、预期结果与贡献", 1)

    add_heading_styled(doc, "4.1 可能的结果模�?, 2)
    add_table(doc,
        ["结果", "理论含义", "后续行动"],
        [
            ["SKD 遗忘显著 > CE 遗忘",
             "历史教师反馈循环确实加剧了信号遗�?,
             "进入 Phase 2：CL 技术适配"],
            ["SKD 遗忘 �?CE 遗忘",
             "α_t 混合已提供足够保护，遗忘是训练的通病而非 SKD 特有",
             "分析为什�?α_t 混合有效，提�?α_t 调度建议"],
            ["SKD 遗忘 < CE 遗忘（意外）",
             "历史教师�?EMA 平滑提供了抗遗忘正则�?,
             "深入分析机制，可能成�?SKD 的新优势论证"],
            ["遗忘�?α_t < 0.5 后加�?,
             "Wu 理论的直接验证：α_t 存在安全下界",
             "提出 α_t 不应低于 0.5 的设计原�?],
        ],
        col_widths=[4.0, 5.0, 5.0]
    )

    add_heading_styled(doc, "4.2 核心贡献", 2)
    add_body(doc,
        "�?）首次将 Wu (ICML 2026) 的迭代自训练遗忘理论应用到真实深度网�?SKD 训练中，"
        "使用 Stern �?Forget Fraction 度量提供系统的实验验证�?
    )
    add_body(doc,
        "�?）首次从持续学习视角审视 SKD 中的遗忘问题，建立了两个领域的概念桥梁—�?
        "SKD 的内部遗忘与 CL 的跨任务遗忘在本质上的相似性�?
    )
    add_body(doc,
        "�?）提出教师置信度加权的抗遗忘约束——这�?CL 社区�?SKD 社区"
        "均未探索的技术方向，具有跨领域的新颖性�?
    )
    add_body(doc,
        "�?）无�?Phase 1 的结果如何，都提供了有价值的知识�?
        "�?SKD 遗忘严重 �?提出了需要解决的新问题；"
        "�?SKD 遗忘不严�?�?解释了为什�?α_t 机制能有效防止遗忘�?
    )

    add_heading_styled(doc, "4.3 发表潜力", 2)
    add_table(doc,
        ["场景", "发表潜力", "可能的目�?],
        [
            ["SKD 遗忘显著 + CL 技术修复有�?, "⭐⭐⭐⭐ �?, "NeurIPS / ICML / ICLR"],
            ["SKD 遗忘显著 + CL 技术修复部分有�?, "⭐⭐�?中高", "CVPR / AAAI / IJCAI"],
            ["SKD 遗忘不显著（中�?否定发现�?, "⭐⭐ �?, "TMLR / 技术报�?],
        ],
        col_widths=[5.5, 2.5, 5.0]
    )

    # ============ REFERENCES ============
    doc.add_page_break()
    add_heading_styled(doc, "参考文�?, 1)

    refs = [
        "[1] Wu M, Yang A Y, Sun Q. Why Self-Training Helps and Hurts: Denoising vs. Signal "
        "Forgetting[C]. International Conference on Machine Learning (ICML), 2026. "
        "[核心理论：迭代自训练�?U 型风险曲线，信号遗忘与去噪的权衡。]",

        "[2] Stern U, Yaacoby T, Weinshall D. On Local Overfitting and Forgetting in Deep "
        "Neural Networks[C]. AAAI Conference on Artificial Intelligence (AAAI), 2025. "
        "[核心度量：Forget Fraction 的严格定义和计算方法。]",

        "[3] Stern U, Corn B, Weinshall D. Forget Me Not: Fighting Local Overfitting with "
        "Knowledge Fusion and Distillation[J]. IEEE TPAMI, 2026. "
        "[修复方法：Checkpoint 融合 + 蒸馏对抗局部过拟合。]",

        "[4] Kirkpatrick J, et al. Overcoming Catastrophic Forgetting in Neural Networks[J]. "
        "PNAS, 2017, 114(13): 3521-3526. [EWC：基�?Fisher 信息的参数重要性约束。]",

        "[5] Aljundi R, et al. Memory Aware Synapses: Learning What (Not) to Forget[C]. "
        "ECCV, 2018: 139-154. [MAS：基于参数敏感度的抗遗忘方法。]",

        "[6] Li Z, Hoiem D. Learning without Forgetting[J]. IEEE TPAMI, 2017, 40(12): "
        "2935-2947. [LwF：旧模型输出作为软目标。与 SKD 的机制高度相似。]",

        "[7] Lopez-Paz D, Ranzato M A. Gradient Episodic Memory for Continual Learning[C]. "
        "NeurIPS, 2017. [GEM：梯度方向约束防止遗忘。]",

        "[8] Das R, Sanghavi S. Understanding Self-Distillation in the Presence of Label "
        "Noise[C]. ICML, 2023: 7102-7140. [自蒸馏在标签噪声下的理论分析。]",

        "[9] Yan H, Wang L, Ma K, Zhong Y. Orchestrate Latent Expertise: Advancing Online "
        "Continual Learning with Multi-Level Supervision and Reverse Self-Distillation[C]. "
        "CVPR, 2024. [MOSE：CL 中的反向自蒸馏——CL 使用 SD 防遗忘。]",

        "[10] Kim K, Ji B, Yoon D, Hwang S. Self-Knowledge Distillation with Progressive "
        "Refinement of Targets[C]. ICCV, 2021. [PS-KD：渐进式自知识蒸馏。]",

        "[11] Li Z, Li X, Yang L, Song R, Yang J, Pan Z. Dual Teachers for Self-Knowledge "
        "Distillation[J]. Pattern Recognition, 2024, 151: 110422. [DTSKD 原始论文。]",

        "[12] Shen Y, et al. Self-Distillation from the Last Mini-Batch for Consistency "
        "Regularization[C]. CVPR, 2022. [DLB：上一 mini-batch 自蒸馏。]",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(ref)
        r.font.name = 'Times New Roman'; r._element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋')
        r.font.size = Pt(9.5)

    # ============ SAVE ============
    output_path = r"D:\Python_code\DTSKD\SKD遗忘问题_开题报�?docx"
    doc.save(output_path)
    print(f"开题报告已保存至：{output_path}")


if __name__ == '__main__':
    create_report()
