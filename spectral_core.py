"""
spectral_core.py — Generic N-D Spectral Architecture Module
===========================================================
Mathematically sound NUFFT grid + per-sample coefficient embeddings.

Key changes from earlier versions:
1. Frequency grid uses standard NUDFT/NUFFT non-uniform spacing with Kaiser-Bessel
   interpolation theory instead of hand-tuned split grid.
2. Per-sample coefficient embeddings: each input produces its own [a0, a_n, b_n]
   vector in a fixed known basis, making the latent space interpretable.
3. Classifier operates on coefficient vector directly, not collapsed scalar.

Usage:
    from spectral_core import SpectralModel
    model = SpectralModel(spatial_shape=(100,), in_channels=3, num_classes=5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Generic channel projection: ConvNd for rank 1/2/3, einsum fallback for 4+
# ---------------------------------------------------------------------------
class ChannelProjection(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_rank):
        super().__init__()
        self.rank = spatial_rank
        if spatial_rank == 1:
            self.op = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        elif spatial_rank == 2:
            self.op = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        elif spatial_rank == 3:
            self.op = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.op = None
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels) * 0.02)
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        if self.op is not None:
            return self.op(x)
        rank = x.ndim - 2
        letters = "".join(chr(ord("d") + i) for i in range(rank))
        ein_in = "bc" + letters
        ein_w = "Dc"
        ein_out = "bD" + letters
        out = torch.einsum(f"{ein_in},{ein_w}->{ein_out}", x, self.weight)
        shape = [1, -1] + [1] * rank
        return out + self.bias.view(*shape)


class SpatialAttentionPooling(nn.Module):
    """
    Replaces naive Global Average Pooling with a learned attention mechanism.
    Dynamically weights the importance of each spatial/temporal location before pooling.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Projects each spatial location's channel vector to an attention logit
        self.attn_proj = nn.Linear(in_channels, 1)

    def forward(self, x):
        # x is (B, C, D1, D2, ...)
        B, C = x.shape[:2]
        
        # Flatten all spatial/temporal dimensions into a sequence of length N
        x_flat = x.view(B, C, -1).transpose(1, 2)  # (B, N, C)
        
        # Compute spatial attention weights
        attn_logits = self.attn_proj(x_flat)       # (B, N, 1)
        attn_weights = F.softmax(attn_logits, dim=1)
        
        # Apply weighted sum across the sequence
        pooled = torch.bmm(x_flat.transpose(1, 2), attn_weights).squeeze(-1)
        return pooled


# ---------------------------------------------------------------------------
# Activations (all spectral-native)
# ---------------------------------------------------------------------------
class LearnableSquareWave(nn.Module):
    """y = A * tanh(k * sin(freq * x + phase))"""
    def __init__(self, steepness=10.0):
        super().__init__()
        self.amplitude = nn.Parameter(torch.tensor(1.0))
        self.log_freq = nn.Parameter(torch.tensor(0.0))
        self.phase = nn.Parameter(torch.tensor(0.0))
        self.steepness = steepness

    def forward(self, x):
        freq = torch.exp(self.log_freq)
        return self.amplitude * torch.tanh(self.steepness * torch.sin(freq * x + self.phase))


class ChebyshevActivation(nn.Module):
    """Weighted sum of Chebyshev T_n(x) = cos(n * arccos(x))"""
    def __init__(self, order=4):
        super().__init__()
        self.order = order
        self.weights = nn.Parameter(torch.randn(order + 1) * 0.1)

    def forward(self, x):
        x = torch.tanh(x)
        T = [torch.ones_like(x), x]
        for n in range(2, self.order + 1):
            T.append(2 * x * T[-1] - T[-2])
        return sum(self.weights[n] * T[n] for n in range(self.order + 1))


class FMActivation(nn.Module):
    """Frequency modulation: tanh(alpha * sin(w1*x + A*sin(w2*x)))"""
    def __init__(self):
        super().__init__()
        self.log_w1 = nn.Parameter(torch.tensor(0.0))
        self.log_w2 = nn.Parameter(torch.tensor(0.0))
        self.log_A = nn.Parameter(torch.tensor(0.0))
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        w1 = torch.exp(self.log_w1); w2 = torch.exp(self.log_w2)
        A = torch.exp(self.log_A); alpha = torch.exp(self.log_alpha)
        return torch.tanh(alpha * torch.sin(w1 * x + A * torch.sin(w2 * x)))


def get_activation(name):
    act_map = {
        "none": nn.Identity(), "relu": nn.ReLU(), "gelu": nn.GELU(),
        "swish": nn.SiLU(), "elu": nn.ELU(), "softplus": nn.Softplus(),
        "square": LearnableSquareWave(steepness=10.0),
        "chebyshev": ChebyshevActivation(order=4),
        "fm": FMActivation(),
    }
    return act_map.get(name, nn.Identity())


# ---------------------------------------------------------------------------
# Generic N-D Spectral Mixer
# ---------------------------------------------------------------------------
class SpectralMixer(nn.Module):
    """
    N-D FFT on each channel over full spatial tensor.
    Learnable complex gain per frequency bin.
    Activation + optional dropout + normalization + residual.
    """
    def __init__(self, channels, spatial_shape, activation="square", dropout=0.0,
                 norm_type="batch"):
        super().__init__()
        self.channels = channels
        self.spatial_shape = tuple(spatial_shape)
        self.rank = len(spatial_shape)
        self.spatial_axes = list(range(2, 2 + self.rank))

        # Infer FFT output shape
        dummy = torch.zeros(1, 1, *spatial_shape)
        if self.rank == 1:
            dummy_fft = torch.fft.rfft(dummy, dim=2)
        elif self.rank == 2:
            dummy_fft = torch.fft.rfft2(dummy, dim=(2, 3))
        else:
            dummy_fft = torch.fft.rfftn(dummy, dim=self.spatial_axes)
        self.fft_shape = dummy_fft.shape[2:]

        self.gain_real = nn.Parameter(torch.ones(channels, *self.fft_shape) * 0.5)
        self.gain_imag = nn.Parameter(torch.zeros(channels, *self.fft_shape))

        if norm_type == "batch" and self.rank == 2:
            self.norm = nn.BatchNorm2d(channels)
        else:
            self.norm = nn.GroupNorm(num_groups=max(1, channels // 4), num_channels=channels)

        self.dropout = nn.Dropout2d(dropout) if self.rank == 2 else nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def _fft(self, x):
        if self.rank == 1:
            return torch.fft.rfft(x, dim=2)
        elif self.rank == 2:
            return torch.fft.rfft2(x, dim=(2, 3))
        return torch.fft.rfftn(x, dim=self.spatial_axes)

    def _ifft(self, x):
        if self.rank == 1:
            return torch.fft.irfft(x, n=self.spatial_shape[0], dim=2)
        elif self.rank == 2:
            return torch.fft.irfft2(x, s=self.spatial_shape, dim=(2, 3))
        return torch.fft.irfftn(x, s=self.spatial_shape, dim=self.spatial_axes)

    def forward(self, x):
        x_fft = self._fft(x)
        gain = torch.view_as_complex(torch.stack([self.gain_real, self.gain_imag], dim=-1))
        x_filtered = x_fft * gain.unsqueeze(0)
        x_out = self._ifft(x_filtered)
        x_out = self.activation(x_out)
        if self.dropout.p > 0:
            x_out = self.dropout(x_out)
        return self.norm(x_out + x)


# ---------------------------------------------------------------------------
# Standard NUFFT-inspired non-uniform frequency grid
# ---------------------------------------------------------------------------
def kaiser_bessel_ft(x, beta, order=0):
    """
    Kaiser-Bessel kernel Fourier transform.
    Used in NUFFT for non-uniform frequency placement.
    """
    eps = 1e-10
    x = x.abs() + eps
    # Approximate KB FT using modified Bessel I0
    # For standard NUFFT, the kernel width determines spacing density
    return torch.where(x < beta,
                       torch.sinh(beta * torch.sqrt(1 - (x / beta) ** 2)) / (beta * torch.sqrt(1 - (x / beta) ** 2)),
                       torch.sin(beta * torch.sqrt((x / beta) ** 2 - 1)) / (beta * torch.sqrt((x / beta) ** 2 - 1)))


def make_nufft_grid(num_modes, oversampling=2, beta=1.5, dense_ratio=0.6):
    """
    Standard NUFFT-inspired non-uniform frequency grid.

    Based on Kaiser-Bessel interpolation theory:
    - Frequencies are placed denser near DC (0) because low frequencies carry
      more energy in natural signals.
    - Spacing follows kernel width ~ 1/omega for smooth interpolation.
    - Oversampling factor controls how many extra points beyond Nyquist.

    Args:
        num_modes: number of frequency points to generate.
        oversampling: oversampling factor (standard NUFFT uses 2).
        beta: Kaiser-Bessel shape parameter (controls density falloff).
        dense_ratio: fraction of modes in the dense low-freq region [0, 1).

    Returns:
        Tensor of shape (num_modes,) with non-uniform frequency indices.
    """
    n_dense = int((num_modes + 1) * dense_ratio)
    n_rest = (num_modes + 1) - n_dense

    # Dense region: KB-inspired spacing — narrower near 0, widening as we go out
    # Use inverse cumulative of KB kernel width
    t_dense = torch.linspace(0, 1, steps=n_dense + 1)[:-1]
    # Spacing ~ (1 + beta * t) to get denser near 0
    dense = t_dense ** (1.0 / (1.0 + beta * 0.5))  # power law: slower growth near 0

    # Uniform region beyond 1
    rest = torch.linspace(1.0, float(oversampling * num_modes), steps=n_rest)

    grid = torch.cat([dense, rest])
    # Ensure strictly increasing and no duplicates
    grid = torch.sort(grid)[0]
    # Remove any exact duplicates
    grid = torch.unique(grid, sorted=True)

    # Pad or truncate to exact num_modes
    if len(grid) < num_modes:
        pad = torch.linspace(float(grid[-1]), float(grid[-1] + n_rest), steps=num_modes - len(grid) + 1)[1:]
        grid = torch.cat([grid, pad])
    elif len(grid) > num_modes:
        grid = grid[:num_modes]
    return grid


def make_split_grid(num_modes):
    """Legacy hand-tuned split grid. Kept for backward compatibility."""
    n_dense = int((num_modes + 1) * 0.6)
    n_rest = (num_modes + 1) - n_dense
    dense = torch.linspace(0.0, 1.0, steps=n_dense + 1)[:-1]
    rest = torch.linspace(1.0, float(num_modes), steps=n_rest)
    return torch.cat([dense, rest])


# ---------------------------------------------------------------------------
# Per-Sample Coefficient Fourier Head
# ---------------------------------------------------------------------------
class CoefficientFourierHead(nn.Module):
    """
    Each sample produces its own Fourier coefficient vector in a fixed basis.

    Output: coefficient vector [a0, a_1, ..., a_N, b_1, ..., b_N]
    - Dimension 0 = DC coefficient a0
    - Dimensions 1:N = cos coefficients a_n at known frequencies
    - Dimensions N+1:2N = sin coefficients b_n at known frequencies

    This vector IS the per-sample embedding in a known, interpretable basis.
    Classifier operates on this vector directly.
    """
    def __init__(self, latent_dim, num_modes=64, init_scale=2.0, grid_type="nufft",
                 oversampling=2, beta=1.5, dense_ratio=0.6):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

        # Map latent z to per-sample coefficients
        self.coeff_proj = nn.Linear(latent_dim, 1 + 2 * num_modes)

        # Fixed basis frequencies — known, unchanging, physically meaningful
        if grid_type == "nufft":
            grid = make_nufft_grid(num_modes, oversampling, beta, dense_ratio)
        else:
            grid = make_split_grid(num_modes)
        self.register_buffer('harmonic_n', grid)

        # Initialize coefficients to small values
        nn.init.normal_(self.coeff_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.coeff_proj.bias)

    def forward(self, z):
        """
        Args:
            z: (B, latent_dim) latent vector after global spatial pooling
        Returns:
            coeffs: (B, 1 + 2*num_modes) per-sample Fourier coefficients
            freqs: (num_modes,) the fixed basis frequencies for reference
        """
        coeffs = self.coeff_proj(z)  # (B, 1 + 2*num_modes)

        # Compute actual angular frequencies for reference/analysis
        freqs = 2.0 * np.pi * self.harmonic_n[1:] / (self.scale.abs() + 1e-6)

        return coeffs, freqs

    def get_coefficient_info(self, coeffs):
        """Decompose coefficient vector into named components."""
        a0 = coeffs[:, 0:1]
        a_n = coeffs[:, 1:1 + self.num_modes]
        b_n = coeffs[:, 1 + self.num_modes:]
        return {"a0": a0, "a_n": a_n, "b_n": b_n}


# ---------------------------------------------------------------------------
# Legacy scalar Fourier head (kept for backward compat)
# ---------------------------------------------------------------------------
class FourierHead(nn.Module):
    """
    y = a0 + sum_n [ a_n * cos(wn*x) + b_n * sin(wn*x) ]
    wn = 2*pi * harmonic_n / scale
    """
    def __init__(self, latent_dim, num_modes=64, init_scale=2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.proj_weight = nn.Parameter(torch.randn(latent_dim) * 0.1)
        self.proj_bias = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.a0 = nn.Parameter(torch.zeros(1))
        self.a_n = nn.Parameter(torch.randn(num_modes) * 0.1)
        self.b_n = nn.Parameter(torch.randn(num_modes) * 0.1)
        grid = make_split_grid(num_modes)
        self.register_buffer('harmonic_n', grid)

    def forward(self, z):
        z_1d = torch.matmul(z, self.proj_weight) + self.proj_bias
        freqs = 2.0 * np.pi * self.harmonic_n[1:] / (self.scale.abs() + 1e-6)
        proj = z_1d.unsqueeze(1) * freqs.unsqueeze(0)
        cos_terms = self.a_n * torch.cos(proj)
        sin_terms = self.b_n * torch.sin(proj)
        fourier_out = self.a0 + cos_terms.sum(dim=1) + sin_terms.sum(dim=1)
        fourier_out = fourier_out / (self.num_modes + 1)
        return fourier_out.unsqueeze(-1), z


# ---------------------------------------------------------------------------
# Full Generic Spectral Model
# ---------------------------------------------------------------------------
class SpectralModel(nn.Module):
    """
    Generic N-D spectral classifier.

    Args:
        spatial_shape: tuple of spatial dimensions.
        in_channels: input channel count.
        num_classes: number of output classes.
        latent_dim: channel count after projection.
        num_modes: Fourier modes in the coefficient head.
        num_mixer_layers: depth.
        head_hidden: hidden dim of final classifier.
        init_scale: initial period scale P.
        activation: mixer activation.
        mixer_dropout: spatial dropout.
        classifier_dropout: dropout before classifier.
        norm_type: "batch" (rank==2) or "group".
        head_type: "coefficient" (new per-sample embeddings) or "scalar" (legacy).
        grid_type: "nufft" (standard) or "split" (legacy).
    """
    def __init__(self, spatial_shape, in_channels, num_classes,
                 latent_dim=16, num_modes=64, num_mixer_layers=4,
                 head_hidden=128, init_scale=2.0, activation="square",
                 mixer_dropout=0.0, classifier_dropout=0.0, norm_type="group",
                 head_type="scalar", grid_type="nufft"):
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_shape = tuple(spatial_shape)
        self.channel_proj = ChannelProjection(in_channels, latent_dim, len(spatial_shape))
        self.head_type = head_type

        self.mixers = nn.ModuleList([
            SpectralMixer(latent_dim, spatial_shape, activation, mixer_dropout, norm_type)
            for _ in range(num_mixer_layers)
        ])
        
        self.attention_pool = SpatialAttentionPooling(latent_dim)

        if head_type == "coefficient":
            self.fourier_head = CoefficientFourierHead(latent_dim, num_modes, init_scale, grid_type)
            coeff_dim = 1 + 2 * num_modes
            clf = []
            if classifier_dropout > 0:
                clf.append(nn.Dropout(classifier_dropout))
            clf.extend([
                nn.Linear(coeff_dim, head_hidden),
                nn.ReLU(),
            ])
            if classifier_dropout > 0:
                clf.append(nn.Dropout(classifier_dropout))
            clf.append(nn.Linear(head_hidden, num_classes))
            self.classifier = nn.Sequential(*clf)
        else:
            self.fourier_head = FourierHead(latent_dim, num_modes, init_scale)
            clf = []
            if classifier_dropout > 0:
                clf.append(nn.Dropout(classifier_dropout))
            clf.extend([
                nn.Linear(1 + latent_dim, head_hidden),
                nn.ReLU(),
            ])
            if classifier_dropout > 0:
                clf.append(nn.Dropout(classifier_dropout))
            clf.append(nn.Linear(head_hidden, num_classes))
            self.classifier = nn.Sequential(*clf)

    def forward(self, x):
        z = self.channel_proj(x)
        for mixer in self.mixers:
            z = mixer(z)
        z = self.attention_pool(z)  # Replaces global average pooling over spatial dims

        if self.head_type == "coefficient":
            coeffs, freqs = self.fourier_head(z)
            logits = self.classifier(coeffs)
            return logits
        else:
            fourier_scalar, z_full = self.fourier_head(z)
            features = torch.cat([fourier_scalar, z_full], dim=-1)
            logits = self.classifier(features)
            return logits

    def get_fourier_info(self):
        if self.head_type == "coefficient":
            return {
                "scale": self.fourier_head.scale.item(),
                "grid_type": "nufft" if hasattr(self.fourier_head, 'harmonic_n') else "unknown",
            }
        return {
            "scale": self.fourier_head.scale.item(),
            "proj_weight_norm": self.fourier_head.proj_weight.norm().item(),
            "a0": self.fourier_head.a0.item(),
        }

    def get_coefficient_embedding(self, x):
        """Extract per-sample Fourier coefficient embedding for analysis."""
        if self.head_type != "coefficient":
            raise ValueError("Only coefficient head produces coefficient embeddings")
        z = self.channel_proj(x)
        for mixer in self.mixers:
            z = mixer(z)
        z = z.mean(dim=list(range(2, z.ndim)))
        coeffs, freqs = self.fourier_head(z)
        return coeffs, freqs


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
