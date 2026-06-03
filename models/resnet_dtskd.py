"""
=============================================================================
DTSKD-ResNet: 双教师自知识蒸馏的多分支 ResNet 模型
=============================================================================

【整体架构】
    输入图像 (3×32×32, CIFAR-100)
         │
         ▼
    ┌──────────────────────┐
    │   Backbone ResNet18   │  conv1 → layer1 → layer2 → layer3 → layer4
    │   (标准ResNet基础网络)  │  输出: s1, s2, s3, s4 四个尺度的特征图
    └──────┬───┬───┬───┬───┘
           │   │   │   │
      s1  s2  s3  s4  (通道数: 64 → 128 → 256 → 512)
       │   │   │   │
       │   │   │   └────────────────────────────→ GAP → fc →  backbone_output  (主干预测)
       │   │   │
       │   │   └──── lateral3 ────┐
       │   │                      │  fuse_3 (融合layer3+layer4特征)
       │   │                      ▼
       │   │                    upsample2 (自顶向下传播)
       │   │                      │
       │   └──── lateral2 ────────┤
       │                          │  fuse_2 (融合layer2+高层特征)
       │                          ▼
       │                        upsample1
       │                          │
       └──── lateral1 ────────────┤
                                  │  fuse_1 (融合layer1+高层特征)
                                  ▼
                                 fc_b1 → b1_output  (浅层分支预测)
                                 fc_b2 → b2_output  (中层分支预测)
                                 fc_b3 → b3_output  (深层分支预测)

【FPN-like 特征融合说明 (自顶向下 + 横向连接)】
    lateral: 从 backbone 各层提取特征，用 AdaptiveAvgPool + 1×1 Conv 压缩到 512 维
    upsample: 自顶向下传播，用 1×1 Conv + BN + ReLU
    fuse: 将 lateral 的浅层特征 与 upsample 的高层特征做 concat，再用 1×1 Conv 降维

【三个分支的区别】
    branch-1 (最浅, b1): 融合 layer1 特征 + 所有更深层的上采样特征
                          → 感受野最小，关注细节纹理
    branch-2 (中层, b2): 融合 layer2 特征 + layer3/layer4 上采样特征
                          → 感受野中等
    branch-3 (较深, b3): 融合 layer3 特征 + layer4 上采样特征
                          → 感受野最大，关注全局语义

【蒸馏中的角色】
    - backbone_output: 结构教师 (Structural Teacher) — 精度最高，指导各分支学习
    - b1/b2/b3_output: 学生分支 — 互相之间也通过 backbone 间接学习
    - 上一轮 epoch 的各输出: 历史教师 (Historical Teacher) — 提供时序一致性监督

=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['resnet18_dtskd', 'resnet34_dtskd']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3×3 卷积 (不改变尺寸，stride=1 时保持 H×W)"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1×1 卷积 (用于降维/升维，不改变空间尺寸)"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """ResNet 基础残差块 (用于 ResNet18/34)

    结构: x → Conv3×3 → BN → ReLU → Conv3×3 → BN → +x → ReLU
    当 stride≠1 或通道数变化时，shortcut 也通过 1×1 Conv 调整尺寸。
    """
    expansion = 1  # 基础块输出通道倍数 (Bottleneck 为 4)

    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        # shortcut: 当输入输出尺寸不匹配时，用1×1卷积调整
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes))
        self.stride = stride

    def forward(self, x):
        identity = x  # shortcut 连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # 调 整shortcut尺寸

        out += identity  # 残差相加
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet 瓶颈残差块 (用于 ResNet50/101/152)

    结构: x → Conv1×1(降维) → Conv3×3 → Conv1×1(升维) → +x → ReLU
    先降后升，减少计算量。expansion=4 表示最终通道数 = 4×planes。
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)  # 降维
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)  # 3×3卷积
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)  # 升维
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes))
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CIFAR_ResNet(nn.Module):
    """
    =========================================================================
    DTSKD 核心网络: 带 FPN 多分支的 CIFAR ResNet
    =========================================================================

    【网络组成】
      1. Backbone: 标准 ResNet (conv1 + 4 个 layer)
      2. Lateral 连接: 从 backbone 各层抽取 512 维特征
      3. Upsample 层: 自顶向下传播高层语义特征
      4. Fuse 层: 融合 lateral(浅层细节) + upsample(高层语义) → 1×1 Conv 降维
      5. 四个分类头: fc (backbone), fc_b1, fc_b2, fc_b3

    【创建方式】
      resnet18_dtskd(num_classes=100) → CIFAR_ResNet(BasicBlock, [2,2,2,2])
      resnet34_dtskd(num_classes=100) → CIFAR_ResNet(BasicBlock, [3,4,6,3])
    """

    def __init__(self, block, num_blocks, num_classes=100, bias=False):
        """
        Args:
            block: 残差块类型 (BasicBlock 或 Bottleneck)
            num_blocks: 每层残差块数量，如 [2,2,2,2] (ResNet18)
            num_classes: 分类数量 (CIFAR-100 = 100)
        """
        super(CIFAR_ResNet, self).__init__()
        self.in_planes = 64

        # ===== Backbone: 标准 ResNet 主干 =====
        # 注意: CIFAR 图像小 (32×32)，第一个卷积用 3×3 stride=1 (不用 ImageNet 的 7×7 stride=2)
        self.conv1 = conv3x3(3, 64)  # 输入: 3×32×32 → 输出: 64×32×32
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 每个 layer 输出通道数: [64, 128, 256, 512] × expansion
        self.network_channels = [
            64 * block.expansion,    # layer1 输出: 64
            128 * block.expansion,   # layer2 输出: 128
            256 * block.expansion,   # layer3 输出: 256
            512 * block.expansion,   # layer4 输出: 512
        ]

        # 四个 layer: 每个由多个残差块堆叠
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)  # 64×32×32
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 128×16×16
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 256×8×8
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 512×4×4

        # ===== Backbone 分类头 =====
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # ===== FPN 横向连接 (Lateral connections) =====
        # 作用: 从 backbone 各层提取特征，压缩到统一的 512 维
        # 结构: AdaptiveAvgPool(1×1) → Conv1×1(通道→512) → BN → ReLU
        laterals, upsample = [], []
        for i in range(3):  # 3个分支，对应 layer1, layer2, layer3
            laterals.append(self._lateral(self.network_channels[i], 512))
        for i in range(1, 4):  # 3个上采样层，从 layer4/layer3/layer2 向下传播
            upsample.append(self._upsample(512, 512))

        self.laterals = nn.ModuleList(laterals)  # 用 ModuleList 确保参数被正确注册
        self.upsample = nn.ModuleList(upsample)

        # ===== Fuse 层: 融合 lateral + upsample 特征 =====
        # 输入: 拼接后 2×512=1024 维 → 1×1 Conv → 512 维
        # fuse_1: 融合 layer1 (最浅层) + 所有高层上采样特征 → branch-1
        self.fuse_1 = nn.Sequential(
            nn.Conv2d(2 * 512 * block.expansion, 512 * block.expansion,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(inplace=True),
        )

        # fuse_2: 融合 layer2 + layer3/layer4 上采样特征 → branch-2
        self.fuse_2 = nn.Sequential(
            nn.Conv2d(2 * 512 * block.expansion, 512 * block.expansion,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(inplace=True),
        )

        # fuse_3: 融合 layer3 + layer4 上采样特征 → branch-3
        self.fuse_3 = nn.Sequential(
            nn.Conv2d(2 * 512 * block.expansion, 512 * block.expansion,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(inplace=True),
        )

        # ===== 三个辅助分支的分类头 =====
        self.fc_b1 = nn.Linear(512, num_classes)  # branch-1 (最浅，细节丰富)
        self.fc_b2 = nn.Linear(512, num_classes)  # branch-2 (中层)
        self.fc_b3 = nn.Linear(512, num_classes)  # branch-3 (较深，语义丰富)

    def _upsample(self, in_channel, out_channel=512):
        """
        自顶向下传播模块
        作用: 将高层特征图调整为与低层相同大小，并统一通道数
        结构: 1×1 Conv → BN → ReLU
        (注意: 这里没有使用插值上采样，因为 CIFAR 特征图尺寸小，直接靠 1×1 Conv)
        """
        layers = []
        layers.append(torch.nn.Conv2d(in_channel, out_channel, kernel_size=1,
                                       stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channel))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _lateral(self, input_size, output_size=512):
        """
        横向连接模块
        作用: 从 backbone 的某一层提取特征，压缩空间维度并统一通道数
        结构: AdaptiveAvgPool2d(1×1) → Conv1×1 → BN → ReLU
        """
        layers = []
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # 将任意尺寸特征图 → 1×1
        layers.append(nn.Conv2d(input_size, output_size, kernel_size=1,
                                 stride=1, bias=False))
        layers.append(nn.BatchNorm2d(output_size))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        构建 ResNet 的一个 layer (由多个残差块堆叠)

        Args:
            block: 残差块类型
            planes: 输出通道数 (未乘 expansion)
            num_blocks: 残差块数量
            stride: 第一个残差块的 stride (用于下采样)
        """
        strides = [stride] + [1] * (num_blocks - 1)  # 只有第一个块做下采样
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  # 更新输入通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 [batch, 3, 32, 32]

        Returns:
            out:    backbone 预测 [batch, 100] — 精度最高，作为结构教师
            t_out1: branch-1 预测 [batch, 100] — 融合最浅层特征
            t_out2: branch-2 预测 [batch, 100] — 融合中层特征
            t_out3: branch-3 预测 [batch, 100] — 融合较深层特征
        """
        # ===== Step 1: Backbone 前向传播 =====
        out = self.conv1(x)       # [B, 3, 32, 32] → [B, 64, 32, 32]
        out = self.bn1(out)
        out = self.relu(out)

        s_out1 = self.layer1(out)  # [B, 64, 32, 32]   ← 最浅层特征，细节丰富
        s_out2 = self.layer2(s_out1)  # [B, 128, 16, 16]
        s_out3 = self.layer3(s_out2)  # [B, 256, 8, 8]
        s_out4 = self.layer4(s_out3)  # [B, 512, 4, 4]  ← 最深层特征，语义丰富

        # ===== Step 2: Backbone 分类 =====
        out = F.adaptive_avg_pool2d(s_out4, (1, 1))  # [B, 512, 4, 4] → [B, 512, 1, 1]
        logits_out = out
        out = out.view(out.size(0), -1)               # [B, 512]
        out = self.fc(out)                            # [B, 100] ← backbone 最终输出

        # ===== Step 3: FPN 自顶向下特征融合 =====
        # t_out4 作为起始点 (最深层特征，不用 lateral，直接用 backbone 的 GAP)
        t_out4 = logits_out  # [B, 512, 1, 1]

        # --- branch-3: fuse_3 = lateral(s_out3) + upsample(t_out4) ---
        upsample3 = self.upsample[2](t_out4)              # [B, 512, 1, 1]
        t_out3 = torch.cat([(upsample3 + self.laterals[2](s_out3)),
                            upsample3], dim=1)            # concat: [B, 1024, 1, 1]
        t_out3 = self.fuse_3(t_out3)                      # → [B, 512, 1, 1]

        # --- branch-2: fuse_2 = lateral(s_out2) + upsample(t_out3) ---
        upsample2 = self.upsample[1](t_out3)              # 从 branch-3 向上传播
        t_out2 = torch.cat([(upsample2 + self.laterals[1](s_out2)),
                            upsample2], dim=1)            # concat: [B, 1024, 1, 1]
        t_out2 = self.fuse_2(t_out2)                      # → [B, 512, 1, 1]

        # --- branch-1: fuse_1 = lateral(s_out1) + upsample(t_out2) ---
        upsample1 = self.upsample[0](t_out2)              # 从 branch-2 向上传播
        t_out1 = torch.cat([(upsample1 + self.laterals[0](s_out1)),
                            upsample1], dim=1)            # concat: [B, 1024, 1, 1]
        t_out1 = self.fuse_1(t_out1)                      # → [B, 512, 1, 1]

        # ===== Step 4: 三个分支分类 =====
        t_out3 = t_out3.view(t_out3.size(0), -1)  # [B, 512]
        t_out3 = self.fc_b3(t_out3)                # [B, 100] ← 最深层分支

        t_out2 = t_out2.view(t_out2.size(0), -1)
        t_out2 = self.fc_b2(t_out2)                # [B, 100] ← 中层分支

        t_out1 = t_out1.view(t_out1.size(0), -1)
        t_out1 = self.fc_b1(t_out1)                # [B, 100] ← 最浅层分支

        return out, t_out1, t_out2, t_out3
        #   ↑backbone  ↑b1      ↑b2      ↑b3
        #   结构教师    浅层学生  中层学生  深层学生


def resnet18_dtskd(pretrained=False, **kwargs):
    """构建 DTSKD-ResNet18: 4层×每层2个BasicBlock = 18层"""
    return CIFAR_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_dtskd(pretrained=False, **kwargs):
    """构建 DTSKD-ResNet34: 4层×每层[3,4,6,3]个BasicBlock = 34层"""
    return CIFAR_ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
