"""
spectral_fnn.py — Fourier Neural Network
========================================
Every neuron is a Fourier series, not just the head.
Customizable layer widths without spatial coupling.

Design:
  FourierLinear(in, out, num_modes):
    For each output neuron j:
      z_j = linear(input)  # scalar per sample
      y_j = a0_j + Σ_n [a_nj·cos(ω_n·z_j) + b_nj·sin(ω_n·z_j)]
    Returns (B, out) vector of scalars.

Stacks like a standard MLP but each hidden unit is a Fourier oscillator.
"""
import torch
import torch.nn as nn
import numpy as np


def make_nufft_grid(num_modes, oversampling=2, beta=1.5, dense_ratio=0.6):
    """Shared NUFFT grid (same as spectral_core)."""
    n_dense = int((num_modes + 1) * dense_ratio)
    n_rest = (num_modes + 1) - n_dense
    t_dense = torch.linspace(0, 1, steps=n_dense + 1)[:-1]
    dense = t_dense ** (1.0 / (1.0 + beta * 0.5))
    rest = torch.linspace(1.0, float(oversampling * num_modes), steps=n_rest)
    grid = torch.cat([dense, rest])
    grid = torch.sort(grid)[0]
    grid = torch.unique(grid, sorted=True)
    if len(grid) < num_modes:
        pad = torch.linspace(float(grid[-1]), float(grid[-1] + n_rest), steps=num_modes - len(grid) + 1)[1:]
        grid = torch.cat([grid, pad])
    elif len(grid) > num_modes:
        grid = grid[:num_modes]
    return grid


class FourierLinear(nn.Module):
    """
    A fully-connected layer where each output neuron applies a Fourier series
    to a learned linear projection of the input.

    Args:
        in_features: dimension of input vector.
        out_features: number of Fourier neurons in this layer.
        num_modes: number of harmonic terms per neuron.
        init_scale: initial period scale P.
    Shape:
        Input:  (B, in_features)
        Output: (B, out_features)
    """
    def __init__(self, in_features, out_features, num_modes=16, init_scale=2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_modes = num_modes

        # Scalar projection per neuron
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Fourier coefficients per neuron
        self.a0 = nn.Parameter(torch.zeros(out_features))
        self.a_n = nn.Parameter(torch.randn(out_features, num_modes) * 0.1)
        self.b_n = nn.Parameter(torch.randn(out_features, num_modes) * 0.1)

        # Per-neuron learnable scale (period P)
        self.scale = nn.Parameter(torch.full((out_features,), init_scale, dtype=torch.float32))

        # Shared fixed frequency grid
        grid = make_nufft_grid(num_modes)
        self.register_buffer('harmonic_n', grid)

    def forward(self, x):
        # x: (B, in_features)
        z = torch.matmul(x, self.weight.t()) + self.bias  # (B, out_features)

        # Angular frequencies per neuron: ω_n = 2π·grid_n / P_j
        # harmonic_n: (num_modes,) ; scale: (out_features,)
        freqs = 2.0 * np.pi * self.harmonic_n.unsqueeze(0) / (self.scale.abs().unsqueeze(1) + 1e-6)
        # freqs: (out_features, num_modes)

        proj = z.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, out_features, num_modes)
        cos_terms = self.a_n.unsqueeze(0) * torch.cos(proj)   # (B, out_features, num_modes)
        sin_terms = self.b_n.unsqueeze(0) * torch.sin(proj)   # (B, out_features, num_modes)
        fourier_out = self.a0.unsqueeze(0) + cos_terms.sum(dim=-1) + sin_terms.sum(dim=-1)

        # Normalize to keep magnitudes comparable to standard linear layers
        return fourier_out / (self.num_modes + 1)


class FourierMLP(nn.Module):
    """
    Stack of FourierLinear layers with customizable widths.

    Args:
        in_features: input dimension (e.g. 784 for flattened 28×28).
        hidden_dims: list of layer widths, e.g. [128, 64, 32].
        num_classes: output class count.
        num_modes: Fourier modes per neuron (default 16).
        dropout: dropout between Fourier layers.
    """
    def __init__(self, in_features, hidden_dims, num_classes, num_modes=16,
                 dropout=0.0, init_scale=2.0):
        super().__init__()
        dims = [in_features] + hidden_dims + [num_classes]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(FourierLinear(dims[i], dims[i+1], num_modes, init_scale))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        # Final classifier is a plain linear (or another FourierLinear)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x may be spatial — flatten first
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
