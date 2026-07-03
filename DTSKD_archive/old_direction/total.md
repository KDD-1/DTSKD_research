 DTSKD 项目全景分析

  一、项目概览

  ┌──────────┬───────────────────────────────────────────────────────────────────┐
  │   项目   │                               说明                                │
  ├──────────┼───────────────────────────────────────────────────────────────────┤
  │ 全称     │ Dual Teachers for Self-Knowledge Distillation（双教师自知识蒸馏） │
  ├──────────┼───────────────────────────────────────────────────────────────────┤
  │ 发表     │ Pattern Recognition 2024，南开大学                                │
  ├──────────┼───────────────────────────────────────────────────────────────────┤
  │ 任务     │ CIFAR-100 图像分类（100类）                                       │
  ├──────────┼───────────────────────────────────────────────────────────────────┤
  │ 框架     │ PyTorch 2.4 + CUDA 12.1                                           │
  ├──────────┼───────────────────────────────────────────────────────────────────┤
  │ 核心思想 │ 用「历史教师」和「结构教师」两个教师来蒸馏一个多分支轻量学生网络  │
  └──────────┴───────────────────────────────────────────────────────────────────┘

  二、项目目录结构

  dtskd/
  ├── main.py                    # 主入口：训练 + 验证流程
  ├── run.txt                    # 运行命令（CPU模式）
  ├── README.md                  # 论文信息 + 使用说明
  ├── models/
  │   ├── network.py             # 网络工厂（根据参数选择模型）
  │   ├── resnet_dtskd.py        # ★ 核心模型：DTSKD-ResNet18/34
  │   ├── resnet.py              # 普通 ResNet（CIFAR）
  │   ├── resnext2_dtskd.py      # DTSKD-ResNeXt
  │   ├── resnext2.py            # 普通 ResNeXt
  │   ├── vgg_dtskd.py           # DTSKD-VGG16
  │   ├── vgg.py                 # 普通 VGG
  │   ├── shufflenetv2_dtskd.py  # DTSKD-ShuffleNetV2
  │   ├── senet.py               # SENet
  │   └── utils.py               # 模型工具
  ├── loader/
  │   ├── custom_dataloader.py   # 数据加载（CIFAR-10/100, ImageNet）
  │   └── custom_datasets.py     # 自定义数据集（返回 index）
  ├── utils/
  │   ├── label_dynamic.py       # ★ KL 散度损失函数
  │   ├── AverageMeter.py        # 平均值/准确率统计
  │   ├── color.py               # 终端彩色输出
  │   ├── custom_transform.py    # 自定义数据增强
  │   ├── dir_maker.py           # 实验目录创建
  │   └── etc.py                 # 进度条、日志、配置保存
  ├── experiments/
  │   └── dtskd_resnet18_test/   # 实验记录
  │       ├── config/config.json # 超参数配置
  │       ├── log/log.txt        # 训练日志（跑了约108个epoch）
  │       ├── checkpoint/        # 模型检查点
  │       └── model/             # 保存的模型
  ├── data/
  │   └── cifar-100-python/      # CIFAR-100 数据集
  ├── tempsave100/               # 抢救下来的训练中间文件
  └── venv/                      # Python 虚拟环境

  三、核心架构：DTSKD-ResNet18

  模型定义在 models/resnet_dtskd.py，这是整个项目最关键的创新点：

  输入图像 (3×32×32)
        │
        ▼
    backbone ResNet18 (conv1 → layer1 → layer2 → layer3 → layer4)
        │                │         │         │         │
        │            s_out1    s_out2    s_out3    s_out4
        │           (64ch)    (128ch)   (256ch)   (512ch)
        │                │         │         │         │
        │                │    lateral2   lateral3      │
        │                │         │         │         │
        │                │         ▼         ▼         │
        │                │      ┌─────────────┐        │
        │                │      │  FPN式特征融合 │      │
        │                │      │ (fuse_1,2,3)  │      │
        │                │      └──┬────┬────┬─┘      │
        │                │         │    │    │         │
        │                ▼         ▼    ▼    ▼         ▼
        │             branch1  branch2 branch3    backbone_head
        │             (fc_b1)  (fc_b2) (fc_b3)    (fc)
        │                │        │       │          │
        ▼                ▼        ▼       ▼          ▼
     output_b1       output_b2 output_b3       output_backbone
     (浅层分支)      (中层分支)  (深层分支)     (主干输出)

  设计要点：
  - Backbone：标准 ResNet18 的 conv1 + 4 个 layer
  - 3 个辅助分支（b1, b2, b3）：通过 FPN 式的横向连接 + 自顶向下融合，从不同深度的特征图生成预测
    - b1（最浅）：融合 layer1 + layer2 上采样特征
    - b2（中层）：融合 layer2 + layer3 上采样特征
    - b3（较深）：融合 layer3 + layer4 上采样特征
  - 最终 4 个输出头：backbone + b1 + b2 + b3，每个都做 100 类分类

  四、训练流程（main.py）

  每个 epoch：
    1. 调整学习率（阶梯衰减，milestone: 150, 225）
    2. 计算 α_t（cosine 衰减，从 0.9 到 0.0）
    3. train()：
       ├─ 从 dataloader 获取 (inputs, targets, indices)
       ├─ 用历史预测 + one-hot 标签生成软目标（soft targets）
       │    soft_target = α_t × one_hot + (1-α_t) × last_epoch_prediction
       ├─ 前向传播：得到 4 组输出（outputs, b1, b2, b3）
       ├─ CE Loss：4 组输出 vs 各自的历史软目标
       ├─ KD Loss：b1/b2/b3 的输出 vs backbone 输出（结构蒸馏）
       │    loss = 0.2×CE + 0.8×KD
       ├─ 反向传播 + AMP 混合精度
       └─ 更新历史预测矩阵（用于下一轮）
    4. val()：
       ├─ 测试 backbone、b1、b2、b3 的 Top-1/Top-5
       └─ 保存最佳模型

  五、双教师机制（论文核心创新）

  ┌───────────────────────┬─────────────────────┬────────────────────┬─────────────────────────────────┐
  │       教师类型        │        来源         │        作用        │            代码实现             │
  ├───────────────────────┼─────────────────────┼────────────────────┼─────────────────────────────────┤
  │ 历史教师 (Historical) │ 上一轮 epoch 的预测 │ 提供时序一致性监督 │ alpha_t 混合历史预测与 one-hot  │
  ├───────────────────────┼─────────────────────┼────────────────────┼─────────────────────────────────┤
  │ 结构教师 (Structural) │ Backbone 主输出     │ 多分支之间互相学习 │ KD loss：b1/b2/b3 对齐 backbone │
  └───────────────────────┴─────────────────────┴────────────────────┴─────────────────────────────────┘

  关键公式（代码中的实现）：
  soft_target = α_t × one_hot_label + (1-α_t) × last_epoch_prediction
  α_t: cosine 衰减，从 0.9 → 0.0（越训练越依赖历史预测）

  六、损失函数（utils/label_dynamic.py）

  KL_Loss(T=4)：  # 用于结构蒸馏
      loss = T² × Σ(teacher_soft × (log(teacher_soft) - student_log_soft))

  KL_Loss2(T=1)： # 用于历史蒸馏（CE 替代）
      loss = T² × Σ(soft_target × (log(soft_target) - student_log_soft))

  七、数据集处理

  loader/custom_datasets.py 的关键设计：自定义 Dataset 返回样本索引，用于在训练时更新对应样本的历史预测矩阵。

  def __getitem__(self, index):
      return img, target, index  # ← 多返回了一个 index

  八、实验状态

  从 log.txt 看，该实验在 CPU 模式下 跑了约 108 个 epoch：
  - Epoch 0: val_top1 = 18.84%
  - Epoch 50: val_top1 ≈ 58~63%（波动较大）
  - Epoch 108: val_top1 ≈ 58.6%

  ▎ ⚠️ 作者在 README 中声明「复现性可能比论文中报告的低一些，因为清理代码时可能误删了一些内容」。另外你在 run.txt 中使用
  ▎ CUDA_VISIBLE_DEVICES=-1 强制 CPU 模式运行，训练速度非常慢且可能影响精度。

  九、架构图总结

  ┌──────────────────────────────────────────────────┐
  │                   DTSKD 训练流程                    │
  ├──────────────────────────────────────────────────┤
  │                                                    │
  │  上一轮预测 ──┐                                    │
  │              ├──→ 软目标生成 ──→ CE Loss          │
  │  one-hot标签 ─┘         │                         │
  │                          ↓                         │
  │  图片 ──→ DTSKD-ResNet ──→ 4组输出               │
  │               │    │    │    │                     │
  │               │    │    │    └──→ backbone预测     │
  │               │    │    └──────→ b3预测 ──┐        │
  │               │    └──────────→ b2预测 ──┤ KD Loss │
  │               └──────────────→ b1预测 ──┘         │
  │                                                    │
  │  Total Loss = 0.2×CE + 0.8×KD                     │
  └──────────────────────────────────────────────────┘ 第二步：修改蒸馏策略 —— 代码改动地图

  整个项目的蒸馏逻辑集中在 main.py 的 3 个位置，改蒸馏只需要动它们：

  改动点 ①：损失函数（main.py 第 475~511 行，train() 函数）

  这是蒸馏怎么做的核心：

  # ===== 当前代码（第 475~511 行）=====
  # CE Loss: 4 个输出头各自 vs 历史软目标
  loss_ce = args.backbone_weight * criterion_CE_hskd(outputs, soft_targets)
  loss_ce += args.b1_weight * criterion_CE_hskd(b1_output, soft_b1_targets)
  loss_ce += args.b2_weight * criterion_CE_hskd(b2_output, soft_b2_targets)
  loss_ce += args.b3_weight * criterion_CE_hskd(b3_output, soft_b3_targets)

  # KD Loss: 3 个分支的输出对齐 backbone（结构蒸馏）
  loss_kd = criterion_KD_hskd(b1_output, outputs)    # b1 学 backbone
  loss_kd += criterion_KD_hskd(b2_output, outputs)   # b2 学 backbone
  loss_kd += criterion_KD_hskd(b3_output, outputs)   # b3 学 backbone

  # 总损失
  loss = args.ce_weight * loss_ce + args.kd_weight * loss_kd

  你想改的几种常见方向：

  ┌──────────────────────────┬────────────────────────────────────────────────────────────────────────┐
  │         改动类型         │                                修改方法                                │
  ├──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ 加一个互学习 loss        │ 添加 loss_mutual = KL(b1, b2) + KL(b2, b3) + KL(b1, b3)                │
  ├──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ 让 backbone 也向分支学习 │ 加 loss_kd += KL(outputs, b1_output) + KL(outputs, b2_output)          │
  ├──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ 换蒸馏温度               │ 改 criterion_KD = KL_Loss(temperature=4) 的 4                          │
  ├──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ 去掉历史蒸馏只用结构蒸馏 │ 设 --HSKD 0                                                            │
  ├──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ 换损失函数               │ 在 utils/label_dynamic.py 里加新的 Loss 类（MSE, Cosine 等）           │
  ├──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ 动态调整分支权重         │ 改 backbone_weight/b1_weight/b2_weight/b3_weight 或让它们随 epoch 变化 │
  └──────────────────────────┴────────────────────────────────────────────────────────────────────────┘

  改动点 ②：软目标生成（main.py 第 440~463 行）

  # ===== 当前：EMA 混合 one-hot 和历史预测 =====
  soft_targets = (alpha_t * targets_one_hot) + ((1 - alpha_t) * all_predictions[input_indices])

  你可以改成：
  - 只用历史预测：soft_targets = all_predictions[input_indices]
  - 只用 one-hot（退化为普通 CE）：soft_targets = targets_one_hot
  - 用其他 teacher 模型的预测：加载外部 teacher 模型的输出
  - 多历史 ensemble：soft_targets = 0.5*last_epoch + 0.3*two_epochs_ago + 0.2*one_hot

  改动点 ③：α_t 衰减策略（main.py 第 330~340 行）

  # ===== 当前：cosine 衰减 =====
  if args.coeff_decay == 'cos':
      ratio = 1.0 * epoch / args.end_epoch
      scale = (math.cos(ratio * PI) + 1.) / 2
      alpha_t = scale * (args.cos_max - args.cos_min) + args.cos_min

  # ===== 当前：linear 衰减 =====
  elif args.coeff_decay == 'linear':
      alpha_t = args.alpha_T * ((epoch + 1) / args.end_epoch)
      alpha_t = 1 - alpha_t

  你可以加自己的策略，比如：
  elif args.coeff_decay == 'exp':
      alpha_t = args.cos_min + (args.cos_max - args.cos_min) * math.exp(-5 * epoch / args.end_epoch)

  ---
  第三步：模型结构改动地图

  如果你要改网络结构（不是蒸馏逻辑），涉及的文件：

  ┌────────────────────┬───────────────────────────────────────────────────────────────────┐
  │       改什么       │                            改哪个文件                             │
  ├────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ ResNet 深度        │ models/resnet_dtskd.py 第 233 行 [2,2,2,2] → [3,4,6,3] (ResNet34) │
  ├────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ 分支数量           │ models/resnet_dtskd.py：删/增 fc_b1/b2/b3，改 forward() 返回值    │
  ├────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ 特征融合方式       │ models/resnet_dtskd.py 的 _lateral(), _upsample(), fuse_*         │
  ├────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ 换 backbone        │ models/network.py：注册新模型 + 写对应的 *_dtskd.py               │
  ├────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ 增加分支间的注意力 │ 在 forward() 的 fuse 操作后加 SE/CA 模块                          │
  └────────────────────┴───────────────────────────────────────────────────────────────────┘
