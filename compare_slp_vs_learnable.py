"""
Compare: Original SLP-style Fourier (integer frequencies + numerical integration)
          vs. Learnable-frequency Fourier (backprop)

This script replicates the user's SLP methodology faithfully:
  - Frequencies are integer harmonics: w_n = 2*pi*n / P
  - Coefficients computed by numerical integration (scipy.integrate.quad)
  - Reconstruction uses exact orthogonal basis

Then compares against the learnable-frequency backprop version on identical
synthetic signals to show why fixed integer frequencies + integration recover
true spectra, while free learnable frequencies drift into correlated modes.
"""

import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn


# =============================================================================
# TARGET SIGNALS (same as visualize_fourier_synthetic.py)
# =============================================================================
def target_multi_freq(x):
    return (
        1.0 * np.sin(2 * np.pi * 0.5 * x) +
        0.5 * np.sin(2 * np.pi * 2.0 * x) +
        0.25 * np.sin(2 * np.pi * 5.0 * x)
    )


def target_square_wave(x, period=4.0):
    return np.sign(np.sin(2 * np.pi * x / period))


def target_chirp(x):
    return np.sin(2 * np.pi * (0.5 * x + 2.0 * x**2))


# =============================================================================
# SLP-STYLE: fixed integer frequencies + numerical integration
# =============================================================================
class SLP_FourierSeries:
    """
    Exact replica of user's compute_real_fourier_coeffs + reconstruction.
    Frequencies are integer harmonics of fundamental 2*pi/P.
    """
    def __init__(self, func, N, P):
        self.N = N
        self.P = P
        self.coeffs = self._compute_coeffs(func)

    def _compute_coeffs(self, func):
        result = []
        for n in range(self.N + 1):
            an = (2. / self.P) * spi.quad(
                lambda t: func(t) * np.cos(2 * np.pi * n * t / self.P), 0, self.P
            )[0]
            bn = (2. / self.P) * spi.quad(
                lambda t: func(t) * np.sin(2 * np.pi * n * t / self.P), 0, self.P
            )[0]
            result.append((an, bn))
        return np.array(result)

    def predict(self, t):
        result = np.zeros_like(t, dtype=np.float64)
        A = self.coeffs[:, 0]
        B = self.coeffs[:, 1]
        for n in range(0, len(self.coeffs)):
            if n > 0:
                result += A[n] * np.cos(2. * np.pi * n * t / self.P) + \
                          B[n] * np.sin(2. * np.pi * n * t / self.P)
            else:
                result += A[0] / 2.
        return result

    def get_spectral_table(self):
        A = self.coeffs[:, 0]
        B = self.coeffs[:, 1]
        energy = np.sqrt(A**2 + B**2)
        table = []
        for n in range(self.N + 1):
            table.append({
                "mode": n,
                "frequency_hz": float(n / self.P),
                "frequency_rad": float(2 * np.pi * n / self.P),
                "A_cos": float(A[n]),
                "B_sin": float(B[n]),
                "energy": float(energy[n]),
                "phase_rad": float(np.arctan2(-B[n], A[n])) if n > 0 else 0.0,
            })
        # sort by energy descending
        table.sort(key=lambda x: -x["energy"])
        return {"dc": float(A[0] / 2), "modes": table}


# =============================================================================
# LEARNABLE: free-frequency backprop model (from visualize_fourier_synthetic.py)
# =============================================================================
class LearnableFourierRegressor(nn.Module):
    def __init__(self, num_modes=32):
        super().__init__()
        self.num_modes = num_modes
        self.freqs = nn.Parameter(torch.randn(num_modes) * 0.5)
        self.A = nn.Parameter(torch.randn(num_modes) * 0.1)
        self.B = nn.Parameter(torch.randn(num_modes) * 0.1)
        self.dc = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        proj = x * self.freqs.unsqueeze(0)
        out = self.dc + torch.sum(
            self.A * torch.cos(proj) + self.B * torch.sin(proj), dim=-1
        )
        return out

    def get_spectral_table(self):
        freqs = self.freqs.detach().cpu().numpy()
        A = self.A.detach().cpu().numpy()
        B = self.B.detach().cpu().numpy()
        energy = np.sqrt(A**2 + B**2)
        idx = np.argsort(-energy)
        table = []
        for i in idx:
            table.append({
                "mode": int(i),
                "frequency_rad": float(freqs[i]),
                "frequency_hz": float(freqs[i]) / (2 * np.pi),
                "A_cos": float(A[i]),
                "B_sin": float(B[i]),
                "energy": float(energy[i]),
                "phase_rad": float(np.arctan2(-B[i], A[i])),
            })
        return {"dc": float(self.dc.item()), "modes": table}


def train_learnable(func, x_train, y_train, num_modes=32, epochs=2000, lr=1e-2):
    x_t = torch.from_numpy(x_train).unsqueeze(-1)
    y_t = torch.from_numpy(y_train)
    model = LearnableFourierRegressor(num_modes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_t)
        loss = criterion(pred, y_t)
        loss.backward()
        opt.step()
    return model


# =============================================================================
# Comparison experiment on one target
# =============================================================================
def run_comparison(target_name, target_fn, P, N_slp, x_test, y_test, num_learnable_modes=64):
    print(f"\n{'='*70}")
    print(f"Target: {target_name} | Period/domain P={P}")
    print(f"SLP modes: integer 0..{N_slp} | Learnable modes: {num_learnable_modes}")
    print('='*70)

    # --- SLP numeric integration ---
    slp = SLP_FourierSeries(target_fn, N_slp, P)
    y_slp = slp.predict(x_test)
    mse_slp = np.mean((y_slp - y_test)**2)

    # --- Train learnable ---
    # sample training points
    x_train = np.random.uniform(0, P, size=5000).astype(np.float32)
    y_train = target_fn(x_train).astype(np.float32)
    learn_model = train_learnable(target_fn, x_train, y_train, num_modes=num_learnable_modes, epochs=2000, lr=1e-2)
    learn_model.eval()
    with torch.no_grad():
        y_learn = learn_model(torch.from_numpy(x_test).unsqueeze(-1)).numpy()
    mse_learn = np.mean((y_learn - y_test)**2)

    print(f"SLP   MSE: {mse_slp:.6f}")
    print(f"Learn MSE: {mse_learn:.6f}")

    # --- Spectral tables ---
    slp_table = slp.get_spectral_table()
    learn_table = learn_model.get_spectral_table()

    print("\nSLP Top modes (integer harmonics):")
    for m in slp_table["modes"][:8]:
        print(f"  n={m['mode']:2d} | {m['frequency_hz']:.3f} Hz | energy={m['energy']:.3f} | A={m['A_cos']:7.3f} B={m['B_sin']:7.3f}")

    print("\nLearnable Top modes (free frequencies):")
    for m in learn_table["modes"][:8]:
        print(f"  mode={m['mode']:2d} | {m['frequency_hz']:.3f} Hz (rad={m['frequency_rad']:.3f}) | energy={m['energy']:.3f} | A={m['A_cos']:7.3f} B={m['B_sin']:7.3f}")

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Fits
    ax = axes[0, 0]
    ax.plot(x_test, y_test, label="True", linewidth=2)
    ax.plot(x_test, y_slp, label=f"SLP (int freq, N={N_slp})", linewidth=1.5)
    ax.set_title(f"SLP Fit | MSE={mse_slp:.4f}")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x_test, y_test, label="True", linewidth=2)
    ax.plot(x_test, y_learn, label=f"Learnable (free freq, M={num_learnable_modes})", linewidth=1.5)
    ax.set_title(f"Learnable Fit | MSE={mse_learn:.4f}")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(x_test, y_test, label="True", linewidth=2)
    ax.plot(x_test, y_slp, label="SLP", linewidth=1.2)
    ax.plot(x_test, y_learn, label="Learnable", linewidth=1.2)
    ax.set_title("Overlay Comparison")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Row 2: Spectra
    ax = axes[1, 0]
    energies_slp = [m["energy"] for m in slp_table["modes"]]
    freqs_slp = [m["frequency_hz"] for m in slp_table["modes"]]
    ax.bar(range(len(energies_slp)), energies_slp, color="tab:blue")
    ax.set_title("SLP Integer-Harmonic Energy Spectrum")
    ax.set_xlabel("Mode rank (by energy)"); ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    energies_learn = [m["energy"] for m in learn_table["modes"]]
    freqs_learn = [m["frequency_hz"] for m in learn_table["modes"]]
    ax.bar(range(len(energies_learn)), energies_learn, color="tab:orange")
    ax.set_title("Learnable Free-Frequency Energy Spectrum")
    ax.set_xlabel("Mode rank (by energy)"); ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.scatter(freqs_slp, energies_slp, label="SLP (integer)", s=80, alpha=0.7)
    ax.scatter(freqs_learn, energies_learn, label="Learnable (free)", s=80, alpha=0.7)
    ax.set_title("Frequency vs Energy Scatter")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Energy")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"compare_{target_name}.png"
    plt.savefig(fname, dpi=150)
    print(f"\nSaved plot: {fname}")

    # Save JSONs
    with open(f"compare_{target_name}_slp.json", "w") as f:
        json.dump(slp_table, f, indent=2)
    with open(f"compare_{target_name}_learnable.json", "w") as f:
        json.dump(learn_table, f, indent=2)

    return {
        "target": target_name,
        "mse_slp": float(mse_slp),
        "mse_learnable": float(mse_learn),
        "slp_top_freqs": [m["frequency_hz"] for m in slp_table["modes"][:5]],
        "learnable_top_freqs": [m["frequency_hz"] for m in learn_table["modes"][:5]],
    }


# =============================================================================
# Main
# =============================================================================
def main():
    results = []

    # Target 1: multi_freq (0.5, 2.0, 5.0 Hz mixed signal)
    P1 = 10.0  # one period ~ 2 seconds for 0.5Hz component? No: period of 0.5Hz = 2s
    # Actually use a domain that shows all components cleanly
    x1 = np.linspace(0, 4, 2000).astype(np.float32)  # show 2 cycles of 0.5Hz
    y1 = target_multi_freq(x1)
    results.append(run_comparison("multi_freq", target_multi_freq, P=4.0, N_slp=25, x_test=x1, y_test=y1, num_learnable_modes=64))

    # Target 2: square wave (discontinuous, Gibbs phenomenon expected in SLP)
    P2 = 4.0
    x2 = np.linspace(0, P2, 2000).astype(np.float32)
    y2 = target_square_wave(x2, period=P2)
    results.append(run_comparison("square", lambda t: target_square_wave(t, period=P2), P=P2, N_slp=64, x_test=x2, y_test=y2, num_learnable_modes=128))

    # Target 3: chirp (non-stationary, both should struggle)
    x3 = np.linspace(0, 2, 2000).astype(np.float32)
    y3 = target_chirp(x3)
    results.append(run_comparison("chirp", target_chirp, P=2.0, N_slp=40, x_test=x3, y_test=y3, num_learnable_modes=128))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        print(f"\n{r['target']}:")
        print(f"  SLP MSE       = {r['mse_slp']:.6f}")
        print(f"  Learnable MSE = {r['mse_learnable']:.6f}")
        print(f"  SLP top freqs       = {r['slp_top_freqs']}")
        print(f"  Learnable top freqs = {r['learnable_top_freqs']}")


if __name__ == "__main__":
    main()
