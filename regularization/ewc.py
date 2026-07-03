"""
EWC (Elastic Weight Consolidation) for SKD

EWC 在参数空间施加二次约束: L_EWC = (λ/2) · Σ_i F_i · (θ_i - θ_i*)^2
用于保护在验证集上正确分类的样本对应的"已知正确知识"不被遗忘。

使用方式:
    ewc = EWCRegularizer(ewc_lambda=0.5, ewc_freq=5)
    for epoch in range(...):
        # epoch 内: 自动选择是否更新 Fisher
        ewc.on_epoch_end(net, valid_loader, criterion_ce, args, per_sample_correct)
        # 训练 step 内:
        loss_ewc = ewc.compute_loss(net)
"""

import torch
import torch.nn as nn


class EWCRegularizer:
    """EWC 正则化器。

    参数:
        ewc_lambda: EWC 约束强度 (0 = 禁用)
        ewc_freq:   每 N 个 epoch 更新一次 Fisher 信息
    """

    def __init__(self, ewc_lambda: float = 0.0, ewc_freq: int = 5):
        self.ewc_lambda = ewc_lambda
        self.ewc_freq = max(1, ewc_freq)
        self.ref_params: dict = {}   # {name: θ*}
        self.fisher: dict = {}       # {name: F_i}

    @property
    def enabled(self) -> bool:
        return self.ewc_lambda > 0.0

    def compute_loss(self, net: nn.Module) -> 'torch.Tensor | None':
        """计算 EWC 约束损失。

        返回: L_EWC = (λ/2)·Σ F_i·(θ_i - θ_i*)^2, 或 None (未初始化时)
        """
        if not self.enabled or not self.fisher or not self.ref_params:
            return None

        loss = 0.0
        for name, param in net.named_parameters():
            if name in self.fisher and name in self.ref_params:
                loss += (self.fisher[name] * (param - self.ref_params[name]) ** 2).sum()

        return 0.5 * self.ewc_lambda * loss

    def on_epoch_end(self, net: nn.Module, valid_loader, criterion_ce,
                     args, per_sample_correct, epoch: int):
        """更新 Fisher 信息和参考参数 (按 ewc_freq 频率)。

        仅当 (1) EWC 启用, (2) epoch % ewc_freq == 0, (3) epoch >= ewc_freq 时更新。
        """
        if not self.enabled:
            return
        if epoch % self.ewc_freq != 0 or epoch < self.ewc_freq:
            return

        correct_indices = per_sample_correct.nonzero(as_tuple=True)[0]
        if len(correct_indices) == 0:
            return

        self.fisher = _compute_fisher(
            net, valid_loader, criterion_ce, args,
            num_samples=min(2000, len(correct_indices)))
        self.ref_params = _snapshot_params(net)

    def state_dict(self) -> dict:
        return {'ref_params': {k: v.clone() for k, v in self.ref_params.items()},
                'fisher': {k: v.clone() for k, v in self.fisher.items()}}

    def load_state_dict(self, sd: dict):
        if 'ref_params' in sd:
            self.ref_params = {k: v.clone() for k, v in sd['ref_params'].items()}
        if 'fisher' in sd:
            self.fisher = {k: v.clone() for k, v in sd['fisher'].items()}


# ── 辅助函数 ──────────────────────────────────────────

def _snapshot_params(net: nn.Module) -> dict:
    """保存当前参数作为 EWC 参考点 θ*"""
    ref = {}
    for name, param in net.named_parameters():
        ref[name] = param.data.clone()
    return ref


def _compute_fisher(net: nn.Module, dataloader, criterion_ce,
                    args, num_samples: int = 2000) -> dict:
    """计算 Fisher 信息矩阵对角线。

    F_i = E_x[(∂L_CE/∂θ_i)^2], 仅对当前正确分类的样本累积。
    返回: {name: fisher_diagonal_tensor}
    """
    net.eval()
    fisher = {}
    for name, param in net.named_parameters():
        fisher[name] = torch.zeros_like(param)

    count = 0
    for inputs, targets, _ in dataloader:
        if count >= num_samples:
            break
        if args.gpu is not None:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        net.zero_grad()
        outputs, _, _, _ = net(inputs)
        loss = criterion_ce(outputs, targets)
        loss.backward()

        _, predicted = outputs.max(1)
        correct_mask = (predicted == targets).float()

        for name, param in net.named_parameters():
            if param.grad is not None:
                grad_sq = param.grad.data ** 2
                weight = correct_mask.mean()
                fisher[name] += grad_sq * weight

        count += inputs.size(0)

    # 归一化
    num_batches = max(count / inputs.size(0), 1)
    for name in fisher:
        fisher[name] /= num_batches

    net.train()
    return fisher
