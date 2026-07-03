"""正则化模块 — 抗遗忘约束的统一接口"""

import torch.nn as nn

class BaseRegularizer:
    """正则化基类 — 所有抗遗忘方法继承此类。

    设计原则：
      - compute_loss() 在训练 step 中每 batch 调用
      - on_epoch_begin/end() 可选 hook，用于更新内部状态
      - state_dict/load_state_dict 用于 checkpoint 存取
    """

    def on_epoch_begin(self, epoch: int):
        pass

    def compute_loss(self, net: nn.Module, **kwargs) -> 'torch.Tensor | None':
        """计算正则化损失。返回 None 表示不施加约束。"""
        raise NotImplementedError

    def on_epoch_end(self, epoch: int):
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, sd: dict):
        pass

from .mcw_af import MCW_AF
from .ewc import EWCRegularizer
