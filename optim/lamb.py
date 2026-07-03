"""
LAMB Optimizer (Layer-wise Adaptive Moments for Batch training)

Ref: You et al., "Large Batch Optimization for Deep Learning", ICLR 2020
"""

import torch


class LAMB(torch.optim.Optimizer):
    """LAMB optimizer with decoupled weight decay and layer-wise adaptive LR."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, adam_mode=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam_mode = adam_mode
        super(LAMB, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                t = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                step_size = group['lr']
                if not group['weight_decay']:
                    step_size /= bias_correction1
                adam_step = exp_avg / bias_correction1
                adam_step.div_(exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(group['eps']))

                if group['weight_decay']:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                weight_norm = p.norm(2)
                adam_norm = adam_step.norm(2)

                if weight_norm > 0 and adam_norm > 0:
                    trust_ratio = weight_norm / adam_norm
                else:
                    trust_ratio = 1.0

                if self.adam_mode:
                    trust_ratio = 1.0

                p.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss
