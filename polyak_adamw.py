"""
PolyakAdamW – AdamW with Polyak step-size damping.
Correct formula: scale = loss / ||grad|| (not squared),
so that lr * scale matches SGD-Polyak step magnitude
when applied to AdamW's normalized direction m/sqrt(v).

Modes:
  damping='global'  – single scale  loss/||g_total||
  damping='group'   – per-param-group scale  loss/||g_i||
  damping='none'    – pure AdamW

Usage:
    optimizer = PolyakAdamW(params, lr=0.25, damping='group')
    loss.backward()
    optimizer.step(loss=loss)
"""
import torch
from torch.optim import Optimizer
import math

class PolyakAdamW(Optimizer):
    def __init__(self, params, lr=0.25, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-4, damping='group', max_scale=None,
                 log_scale=False):
        if lr < 0.0: raise ValueError(f"lr {lr}")
        if damping not in ('global', 'group', 'none'):
            raise ValueError("damping must be 'global', 'group', or 'none'")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, correct_bias=True)
        super().__init__(params, defaults)
        self.damping = damping
        self.max_scale = max_scale
        self.log_scale = log_scale
        self.last_scales = {}
        self._loss_buffer = []  # accumulate detached loss tensors across grad-accum steps

    def append_loss(self, loss):
        """Called every training_step to accumulate loss tensors (no .item() sync)."""
        self._loss_buffer.append(loss.detach() if isinstance(loss, torch.Tensor) else torch.tensor(loss))

    @torch.no_grad()
    def step(self, closure=None, *, loss=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if loss is None:
            if not self._loss_buffer:
                raise ValueError(
                    "PolyakAdamW.step() needs `loss=loss_tensor` or call append_loss() before step()"
                )
            # Average accumulated losses (consistent with HF Trainer's scaled backward)
            loss = sum(self._loss_buffer) / len(self._loss_buffer)
            self._loss_buffer = []

        # ---- Polyak damping: scale = loss / ||grad|| (not squared) ----
        if self.damping == 'none':
            scales = [1.0] * len(self.param_groups)
        elif self.damping == 'global':
            grad_sq = torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)
            for g in self.param_groups:
                for p in g["params"]:
                    if p.grad is not None:
                        grad_sq.add_(p.grad.pow(2).sum())
            grad_norm = grad_sq.sqrt().clamp_(min=1e-8)
            scale = (loss / grad_norm).item()
            if self.max_scale is not None:
                scale = min(scale, self.max_scale)
            scales = [scale] * len(self.param_groups)
            self.last_scales = {-1: scale}
        else:  # 'group'
            scales = []
            for i, g in enumerate(self.param_groups):
                grad_sq = torch.tensor(0.0, device=g["params"][0].device)
                for p in g["params"]:
                    if p.grad is not None:
                        grad_sq.add_(p.grad.pow(2).sum())
                grad_norm = grad_sq.sqrt().clamp_(min=1e-8)
                scale = (loss / grad_norm).item()
                if self.max_scale is not None:
                    scale = min(scale, self.max_scale)
                scales.append(scale)
                self.last_scales[i] = scale

        if self.log_scale:
            for i, s in enumerate(scales):
                print(f"    [PolyakAdamW group={i}] scale={s:.6f}")

        # ---- AdamW step with Polyak-damped lr ----
        for gi, g in enumerate(self.param_groups):
            orig_lr = g["lr"]
            # The key fix: step magnitude = lr * scale
            # In SGD-Polyak: ||step|| = lr * (loss/||g||²) * ||g|| = lr * loss/||g||
            # In AdamW: direction norm ≈ 1, so we need step_size ≈ lr * loss/||g||
            # Hence scale = loss/||g|| (not squared) makes magnitudes match
            damped_lr = orig_lr * scales[gi]

            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PolyakAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = g['betas']
                state['step'] += 1

                # Decoupled weight decay (uses ORIGINAL lr, not damped)
                if g['weight_decay'] != 0:
                    p.mul_(1 - orig_lr * g['weight_decay'])

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Adam momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(g['eps'])
                step_size = damped_lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
