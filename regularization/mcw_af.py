"""
MCW-AF: Margin-Confidence Weighted Anti-Forgetting SKD

基于边际置信度的选择性抗遗忘正则化。
仅保护高置信度 + 稳定正确的历史知识，不干预低置信度/错误的历史预测。

式(1) 边际置信度:   m_i = p_old(y_i) - max_{c≠y_i} p_old(c)
式(2) 动态权重:     w_i = exp(α·(m_i-τ))  if m_i>0  else  0
式(3) 总约束:       L_AF = λ·(1-α_t)/B · Σ_i w_i · KL(p_old_i || p_new_i)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCW_AF:
    """置信度加权抗遗忘正则化器。

    参数:
        af_lambda:  AF 基础强度 (λ)
        af_alpha:   指数尺度因子 (α) — 越大，高置信度样本保护越强
        af_tau:     中性阈值 (τ) — m_i=τ 时 w_i=1
        num_samples: 训练集样本总数 (用于初始化 stability 向量)
    """

    def __init__(self, af_lambda: float, af_alpha: float, af_tau: float,
                 num_samples: int):
        self.af_lambda = af_lambda
        self.af_alpha = af_alpha
        self.af_tau = af_tau
        # per-sample 预测稳定性 EMA: stability_i ∈ [0,1]
        self.stability = torch.zeros(num_samples, dtype=torch.float32)

    def compute_loss(self,
                     softmax_output: torch.Tensor,
                     targets: torch.Tensor,
                     input_indices: torch.Tensor,
                     all_predictions: torch.Tensor,
                     alpha_t: float,
                     epoch: int) -> 'torch.Tensor | None':
        """计算 MCW-AF 损失。

        参数:
            softmax_output:  当前模型 softmax 输出 [B, C]
            targets:         GT 标签 [B]
            input_indices:   样本在全体训练集中的索引 [B]
            all_predictions: 上轮历史预测矩阵 [N, C]
            alpha_t:         SKD 的 α_t (GT 混合系数)
            epoch:           当前 epoch (epoch=0 时返回 None)

        返回:
            AF 损失标量 (可 backprop), 或 None (epoch=0 时)
        """
        if epoch <= 0:
            return None

        with torch.no_grad():
            # 更新 per-sample 稳定性 EMA
            correct_mask = (softmax_output.argmax(dim=1) == targets).float().cpu()
            self.stability[input_indices] = (
                0.9 * self.stability[input_indices] + 0.1 * correct_mask
            )
            stab_batch = self.stability[input_indices].cuda()

            p_old_batch = all_predictions[input_indices].cuda()

            # 式(1): 边际置信度
            p_old_correct = p_old_batch[torch.arange(p_old_batch.size(0)), targets]
            p_old_masked = p_old_batch.clone()
            p_old_masked[torch.arange(p_old_masked.size(0)), targets] = -float('inf')
            p_old_max_wrong = p_old_masked.max(dim=1).values
            m = p_old_correct - p_old_max_wrong

            # 式(2): 动态权重 — 抽奖机制 (m_i≤0 → 放弃)
            w = torch.where(
                m > 0,
                torch.exp(self.af_alpha * (m - self.af_tau)),
                torch.zeros_like(m)
            )

        # 前向 KL: KL(p_old || p_new)
        log_p_new = torch.log(softmax_output + 1e-10)
        kl_per_sample = F.kl_div(log_p_new, p_old_batch, reduction='none').sum(dim=1)

        # 式(3): 加权 KL, 时序反转 λ_eff = λ·(1-α_t)
        af_weights = self.af_lambda * (1.0 - alpha_t) * stab_batch * w
        loss_af = (af_weights * kl_per_sample).mean()

        return loss_af

    @property
    def mean_stability(self) -> float:
        return self.stability.mean().item()

    def state_dict(self) -> dict:
        return {'stability': self.stability.clone()}

    def load_state_dict(self, sd: dict):
        if 'stability' in sd:
            self.stability = sd['stability'].clone()
