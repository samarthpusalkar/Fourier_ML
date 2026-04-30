"""
Train a 1D Fourier regression model on synthetic sinusoidal data,
then visualize the learned amplitudes, frequencies, and DC component.

This replaces the original project's numerical integration with backprop,
and lets you inspect what the network "learned" about frequencies.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json


class FourierRegressor(nn.Module):
    """
    Learns a real-valued function on R^1 using a truncated Fourier series:
        f(x) = dc + sum_n [ A_n cos(w_n x) + B_n sin(w_n x) ]
    All parameters (frequencies, amplitudes, DC) are learned via backprop.
    """
    def __init__(self, num_modes=32):
        super().__init__()
        self.num_modes = num_modes

        # Learnable frequencies (radial frequency for each mode)
        self.freqs = nn.Parameter(torch.randn(num_modes) * 0.5)

        # Cosine and sine amplitudes
        self.A = nn.Parameter(torch.randn(num_modes) * 0.1)
        self.B = nn.Parameter(torch.randn(num_modes) * 0.1)

        # DC offset
        self.dc = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch, 1)
        proj = x * self.freqs.unsqueeze(0)  # (batch, num_modes)
        out = self.dc + torch.sum(
            self.A * torch.cos(proj) + self.B * torch.sin(proj),
            dim=-1, keepdim=True
        )
        return out.squeeze(-1)

    def get_spectral_table(self):
        """
        Return a JSON-serializable table of learned parameters sorted by
        energy (sqrt(A^2 + B^2)), so dominant modes are at the top.
        """
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
                "phase_rad": float(np.arctan2(-B[i], A[i])),  # matches standard Fourier phase
            })
        return {
            "dc": float(self.dc.item()),
            "modes": table,
        }


# --- synthetic target functions ---
def target_square_wave(x, period=4.0):
    return np.sign(np.sin(2 * np.pi * x / period))

def target_chirp(x):
    return np.sin(2 * np.pi * (0.5 * x + 2.0 * x**2))

def target_square_plus_noise(x, period=4.0, sigma=0.1):
    return np.sign(np.sin(2 * np.pi * x / period)) + sigma * np.random.randn(*x.shape)

def target_multi_freq(x):
    return (
        1.0 * np.sin(2 * np.pi * 0.5 * x) +
        0.5 * np.sin(2 * np.pi * 2.0 * x) +
        0.25 * np.sin(2 * np.pi * 5.0 * x)
    )


def run_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- generate data ---
    x_train = np.random.uniform(args.xmin, args.xmax, size=args.num_samples).astype(np.float32)
    x_test = np.linspace(args.xmin, args.xmax, args.num_test).astype(np.float32)

    if args.target == "square":
        y_train = target_square_wave(x_train, args.period).astype(np.float32)
        y_test = target_square_wave(x_test, args.period).astype(np.float32)
    elif args.target == "chirp":
        y_train = target_chirp(x_train).astype(np.float32)
        y_test = target_chirp(x_test).astype(np.float32)
    elif args.target == "noisy_square":
        y_train = target_square_plus_noise(x_train, args.period, sigma=0.1).astype(np.float32)
        y_test = target_square_wave(x_test, args.period).astype(np.float32)
    elif args.target == "multi_freq":
        y_train = target_multi_freq(x_train).astype(np.float32)
        y_test = target_multi_freq(x_test).astype(np.float32)
    else:
        raise ValueError(f"Unknown target: {args.target}")

    x_train_t = torch.from_numpy(x_train).unsqueeze(-1)
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test).unsqueeze(-1)
    y_test_t = torch.from_numpy(y_test)

    # --- model ---
    model = FourierRegressor(num_modes=args.num_modes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    losses = []

    # --- training ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if epoch % args.log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_pred = model(x_test_t)
                test_loss = criterion(test_pred, y_test_t)
            print(f"Epoch {epoch:3d} | Train MSE: {loss.item():.6f} | Test MSE: {test_loss.item():.6f}")

    # --- final evaluation ---
    model.eval()
    with torch.no_grad():
        final_pred = model(x_test_t).numpy()

    # --- spectral table ---
    table = model.get_spectral_table()
    with open(args.out_prefix + "_spectral_table.json", "w") as f:
        json.dump(table, f, indent=2)

    # --- plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Fit vs true
    ax = axes[0, 0]
    ax.plot(x_test, y_test, label="True", linewidth=2)
    ax.plot(x_test, final_pred, label="Learned Fourier", linewidth=1.5)
    ax.scatter(x_train[:500], y_train[:500], alpha=0.2, s=5, label="Train samples", color="gray")
    ax.set_title("Learned Fourier Approximation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Training loss
    ax = axes[0, 1]
    ax.semilogy(losses)
    ax.set_title("Training Loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)

    # Plot 3: Magnitude spectrum (energy per mode)
    ax = axes[1, 0]
    freqs = table["modes"]
    energies = [m["energy"] for m in freqs]
    freq_vals = [m["frequency_hz"] for m in freqs]
    colors = plt.cm.viridis(np.linspace(0, 1, len(energies)))
    ax.bar(range(len(energies)), energies, color=colors)
    ax.set_title("Learned Mode Energies (sorted)")
    ax.set_xlabel("Mode rank")
    ax.set_ylabel("Energy = sqrt(A^2 + B^2)")
    ax.grid(True, alpha=0.3)

    # Plot 4: Frequency-vs-energy scatter (the "emerged frequencies")
    ax = axes[1, 1]
    ax.scatter(freq_vals, energies, c=range(len(energies)), cmap="viridis", s=100)
    ax.set_title("Learned Frequencies (Hz) vs Energy")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out_prefix + "_analysis.png", dpi=150)
    print(f"Saved visualization to: {args.out_prefix}_analysis.png")
    print(f"Saved spectral table to: {args.out_prefix}_spectral_table.json")

    # Print top-k modes
    print("\nTop 8 learned modes:")
    for m in table["modes"][:8]:
        print(
            f"  Mode {m['mode']:2d} | freq={m['frequency_rad']:7.3f} rad/s "
            f"({m['frequency_hz']:5.3f} Hz) | "
            f"A={m['A_cos']:7.3f} B={m['B_sin']:7.3f} | "
            f"energy={m['energy']:5.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize learned Fourier parameters on synthetic data")
    parser.add_argument("--target", type=str, default="multi_freq",
                        choices=["square", "chirp", "noisy_square", "multi_freq"])
    parser.add_argument("--num_modes", type=int, default=32,
                        help="Number of Fourier modes (learnable sinusoids)")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--xmin", type=float, default=-4.0)
    parser.add_argument("--xmax", type=float, default=4.0)
    parser.add_argument("--period", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--out_prefix", type=str, default="fourier_synthetic")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
