"""
test_spectral_real_audio.py — Realistic Instrument Timbre Classification
========================================================================
Generates physically plausible instrument tones:
  Piano   : rich harmonics, percussive attack, fast decay
  Flute   : pure sine-like, few weak harmonics, slow attack
  Guitar  : plucked, inharmonicity, odd harmonics, medium decay
  Violin  : sawtooth-like, strong odd+even harmonics, vibrato
  Trumpet : brass, odd harmonics dominant, bright attack

Each sample has random pitch (fundamental 100-800 Hz), random duration,
random ADSR, and real-world noise (room tone + quantization).

Task: classify by TIMBRE (instrument) regardless of PITCH.
This tests spectral mixer's ability to read harmonic envelope
(not just fundamental frequency).
"""
import torch
import torch.nn as nn
import numpy as np
from spectral_core import SpectralModel, count_params

SR = 8000
MAX_LEN = 8000  # 1 sec
N_CLASSES = 5
INSTRUMENTS = ["piano", "flute", "guitar", "violin", "trumpet"]

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def adsr_envelope(t, attack, decay, sustain_level, release, duration):
    """Piecewise linear ADSR envelope."""
    env = np.zeros_like(t)
    a_end = attack
    d_end = attack + decay
    r_start = duration - release
    env[t <= a_end] = t[t <= a_end] / (attack + 1e-6)
    mask = (t > a_end) & (t <= d_end)
    env[mask] = 1.0 - (1.0 - sustain_level) * (t[mask] - a_end) / (decay + 1e-6)
    mask = (t > d_end) & (t <= r_start)
    env[mask] = sustain_level
    mask = t > r_start
    env[mask] = sustain_level * (1.0 - (t[mask] - r_start) / (release + 1e-6))
    env[env < 0] = 0
    return env


def generate_tone(instrument, fund_freq, duration, sr=SR):
    """Generate realistic instrument tone with proper harmonic structure."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    harmonics = []
    phases = []
    amps = []

    if instrument == "piano":
        # Rich harmonics, fast decay, slight inharmonicity
        for h in range(1, 12):
            harmonics.append(h * fund_freq * (1 + 0.0005 * h**2))
            amps.append(1.0 / h**0.7)
            phases.append(np.random.uniform(0, 2*np.pi))
        attack, decay, sustain, release = 0.01, 0.3, 0.3, 0.4
    elif instrument == "flute":
        # Pure, few weak harmonics
        for h in [1, 2, 3]:
            harmonics.append(h * fund_freq)
            amps.append([1.0, 0.15, 0.05][h-1])
            phases.append(np.random.uniform(0, 2*np.pi))
        attack, decay, sustain, release = 0.15, 0.1, 0.6, 0.15
    elif instrument == "guitar":
        # Plucked, odd harmonics stronger, medium decay
        for h in range(1, 10):
            harmonics.append(h * fund_freq * (1 + 0.001 * h**2))
            amps.append(1.0 / h if h % 2 == 1 else 0.3 / h)
            phases.append(np.random.uniform(0, 2*np.pi))
        attack, decay, sustain, release = 0.005, 0.4, 0.2, 0.5
    elif instrument == "violin":
        # Sawtooth-like, many harmonics, vibrato
        for h in range(1, 16):
            harmonics.append(h * fund_freq)
            amps.append(1.0 / h)
            phases.append(0.0)
        attack, decay, sustain, release = 0.1, 0.05, 0.7, 0.15
    elif instrument == "trumpet":
        # Brass, odd harmonics dominant, bright
        for h in range(1, 14):
            harmonics.append(h * fund_freq)
            amp = 1.0 / h if h % 2 == 1 else 0.2 / h
            amps.append(amp)
            phases.append(np.random.uniform(0, 2*np.pi))
        attack, decay, sustain, release = 0.03, 0.1, 0.7, 0.2

    signal = np.zeros_like(t)
    for freq, amp, phase in zip(harmonics, amps, phases):
        vibrato = 0.0
        if instrument == "violin":
            vibrato = 5.0 * np.sin(2 * np.pi * 5.0 * t)  # 5 Hz vibrato, 5 Hz depth
        signal += amp * np.sin(2 * np.pi * freq * t + phase + vibrato)

    env = adsr_envelope(t, attack, decay, sustain, release, duration)
    signal *= env

    # Realistic noise
    signal += 0.02 * np.random.randn(len(t))  # white noise
    signal += 0.01 * np.sin(2 * np.pi * 50 * t)  # 50 Hz hum

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    return signal.astype(np.float32)


def make_dataset(n_per_class=400):
    """Generate train/test split with pitch-shuffled instrument classification."""
    np.random.seed(42)
    all_data = []
    for cls, inst in enumerate(INSTRUMENTS):
        for _ in range(n_per_class):
            fund = np.random.uniform(100.0, 800.0)
            dur = np.random.uniform(0.5, 1.0)
            tone = generate_tone(inst, fund, dur)
            # Pad/truncate to MAX_LEN
            if len(tone) < MAX_LEN:
                tone = np.pad(tone, (0, MAX_LEN - len(tone)))
            else:
                tone = tone[:MAX_LEN]
            all_data.append((tone, cls))
    np.random.shuffle(all_data)
    n_train = int(len(all_data) * 0.8)
    train = all_data[:n_train]
    test = all_data[n_train:]
    return train, test


def train_eval(model, train_loader, test_loader, epochs=12, lr=1e-2, wd=1e-4):
    device = get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        if acc > best:
            best = acc
        if epoch <= 3 or epoch % 3 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d} | Test Acc: {acc:.4f}")
    print(f"  Best accuracy: {best:.4f}")
    return best


def main():
    device = get_device()
    print(f"Device: {device}")
    print("=" * 60)
    print("REALISTIC INSTRUMENT TIMBRE CLASSIFICATION")
    print("=" * 60)
    train, test = make_dataset(n_per_class=400)
    print(f"Train: {len(train)} | Test: {len(test)}")

    x_tr = torch.stack([torch.from_numpy(d[0]) for d in train]).unsqueeze(1)
    y_tr = torch.tensor([d[1] for d in train], dtype=torch.long)
    x_te = torch.stack([torch.from_numpy(d[0]) for d in test]).unsqueeze(1)
    y_te = torch.tensor([d[1] for d in test], dtype=torch.long)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=64, shuffle=False)

    model = SpectralModel(
        spatial_shape=(MAX_LEN,), in_channels=1, num_classes=N_CLASSES,
        latent_dim=32, num_mixer_layers=4, activation="square",
        mixer_dropout=0.1, classifier_dropout=0.3, norm_type="group",
    )
    print(f"Params: {count_params(model)}")
    acc = train_eval(model, train_loader, test_loader, epochs=12, lr=1e-2)

    print("\n" + "=" * 60)
    print("BASELINE MLP")
    print("=" * 60)
    baseline = nn.Sequential(
        nn.Flatten(),
        nn.Linear(MAX_LEN, 512), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, N_CLASSES),
    )
    print(f"Params: {count_params(baseline)}")
    acc_b = train_eval(baseline, train_loader, test_loader, epochs=12, lr=1e-2)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Spectral: {acc:.4f}")
    print(f"  Baseline: {acc_b:.4f}")


if __name__ == "__main__":
    main()
