"""
Test Non-Uniform Frequency Spacing vs Uniform
===============================================

Generate synthetic signals with known frequency content.
Train tiny Fourier models with uniform vs power-law spacing.
Compare MSE on recovery.
"""

import torch
import torch.nn as nn
import numpy as np


class TinyFourierModel(nn.Module):
    """Scalar Fourier series with learnable amplitudes and scale."""
    def __init__(self, num_modes=64, spacing_mode="uniform", alpha=2.0):
        super().__init__()
        self.num_modes = num_modes
        self.amplitude = nn.Parameter(torch.randn(num_modes + 1) * 0.1)
        self.scale = nn.Parameter(torch.tensor(2.0))

        if spacing_mode == "uniform":
            spacing = torch.arange(num_modes + 1, dtype=torch.float32)
        elif spacing_mode == "power":
            n = torch.arange(num_modes + 1, dtype=torch.float32)
            spacing = (n / num_modes) ** alpha * num_modes
        elif spacing_mode == "log":
            c = 1.0
            n = torch.arange(num_modes + 1, dtype=torch.float32)
            spacing = torch.log1p(c * n) / torch.log1p(torch.tensor(c * num_modes)) * num_modes
        else:
            raise ValueError(f"Unknown spacing: {spacing_mode}")
        self.register_buffer('spacing', spacing)

    def forward(self, x):
        # x: (batch,)
        freqs = 2.0 * np.pi * self.spacing / (self.scale.abs() + 1e-6)
        proj = x.unsqueeze(1) * freqs.unsqueeze(0)  # (batch, modes)
        return (self.amplitude * torch.sin(proj)).sum(dim=1)


def generate_signal(t, freq_components):
    """freq_components: list of (amp, freq) tuples"""
    y = torch.zeros_like(t)
    for amp, freq in freq_components:
        y += amp * torch.sin(2 * np.pi * freq * t)
    return y


def train_model(model, t, y_true, epochs=2000, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        y_pred = model(t)
        loss = ((y_pred - y_true) ** 2).mean()
        loss.backward()
        opt.step()
    return loss.item()


def test_case(name, t, y_true, num_modes=64):
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"Modes: {num_modes}")
    print("-"*60)

    for mode, alpha in [("uniform", 1.0), ("power", 1.5), ("power", 2.0), ("log", 1.0)]:
        model = TinyFourierModel(num_modes=num_modes, spacing_mode=mode, alpha=alpha)
        mse = train_model(model, t, y_true)
        print(f"  {mode:8s} (alpha={alpha:.1f}) | MSE: {mse:.8f}")


def main():
    print("Non-Uniform Frequency Spacing Test")
    print("Comparing uniform vs power-law vs log spacing on synthetic signals")

    t = torch.linspace(0, 4.0, 1000)

    # Test 1: Low-frequency dominant signal
    y1 = generate_signal(t, [(2.0, 0.5), (0.5, 1.5), (0.3, 3.0)])
    test_case("Low-freq dominant", t, y1, num_modes=32)

    # Test 2: Signal with a very low frequency + mid frequency
    y2 = generate_signal(t, [(3.0, 0.1), (1.0, 2.0), (0.5, 5.0)])
    test_case("Very low + mid freq", t, y2, num_modes=32)

    # Test 3: Square wave approximation (needs many harmonics)
    y3 = torch.sign(torch.sin(2 * np.pi * 0.5 * t))
    test_case("Square wave (0.5 Hz)", t, y3, num_modes=64)

    # Test 4: Chirp signal (frequency sweeps)
    y4 = torch.sin(2 * np.pi * (0.1 * t + 0.5 * t ** 2))
    test_case("Chirp sweep", t, y4, num_modes=64)

    # Test 5: Signal with closely-spaced low frequencies
    y5 = generate_signal(t, [(1.0, 0.3), (1.0, 0.35), (1.0, 0.4), (0.5, 3.0)])
    test_case("Closely-spaced low freqs", t, y5, num_modes=32)

    print("\n" + "="*60)
    print("Summary: power-law spacing (alpha>1) clusters modes near low")
    print("frequencies, giving finer resolution where precision matters.")
    print("="*60)


if __name__ == "__main__":
    main()
