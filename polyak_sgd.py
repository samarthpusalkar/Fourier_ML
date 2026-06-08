"""
PolyakSGD – drop-in PyTorch optimizer.
Default settings tuned for stable convergence across MLPs, CNNs, and transformers:
  lr=0.25, weight_decay=1e-4, max_scale=10.0
Usage:
    optimizer = PolyakSGD(params, lr=0.25, weight_decay=1e-4)
    loss.backward()
    optimizer.step(loss=loss)
"""
import torch
from torch.optim import Optimizer

class PolyakSGD(Optimizer):
    def __init__(self, params, lr=0.25, weight_decay=1e-4, max_scale=10.0, log_scale=False):
        if lr < 0.0: raise ValueError(f"lr {lr}")
        if weight_decay < 0.0: raise ValueError(f"wd {weight_decay}")
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.max_scale = max_scale
        self.log_scale = log_scale
        self.last_scale = None
        self.last_grad_norm = None

    @torch.no_grad()
    def step(self, closure=None, *, loss=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if loss is None:
            raise ValueError("PolyakSGD.step() needs `loss=loss_tensor` after backward()")

        # ||grad||^2 on-device (single MPS/CPU sync)
        grad_sq = torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    grad_sq.add_(p.grad.pow(2).sum())

        grad_sq.clamp_(min=1e-8)
        self.last_grad_norm = grad_sq.sqrt().item()
        self.last_scale = (loss / grad_sq).item()

        # Safety cap to prevent divergence on large initial loss
        if self.max_scale is not None and self.last_scale > self.max_scale:
            self.last_scale = self.max_scale

        if self.log_scale:
            print(f"    [Polyak] loss={loss.item():.4f} | ||g||={self.last_grad_norm:.4f} | scale={self.last_scale:.6f}")

        for g in self.param_groups:
            lr = g["lr"]
            wd = g["weight_decay"]
            for p in g["params"]:
                if p.grad is None: continue
                if wd != 0:
                    p.grad.add_(p, alpha=wd)
                p.add_(p.grad, alpha=-lr * self.last_scale)
        return loss
