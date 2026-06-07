"""
test_spectral_advanced.py — Real-world-like data tests
=======================================================
1. ECG-like 1D medical signals (normal vs arrhythmia) — tests spectral on physiological data
2. Audio pitch classification (raw waveforms) — tests spectral on sound
3. Spectrogram-as-image (2D freq×time) — tests spectral on time-frequency representations

Uses spectral_core.SpectralModel + MPS.
"""
import torch
import torch.nn as nn
import numpy as np
import os
from spectral_core import SpectralModel, count_params


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


# ---------------------------------------------------------------------------
# Test 1: ECG-like 1D signals (normal vs arrhythmia)
# ---------------------------------------------------------------------------
def generate_ecg_beat(t, cls, noise=0.05):
    """
    cls=0: normal sinus rhythm (clean P-QRS-T)
    cls=1: arrhythmia (irregular spacing, extra peaks)
    """
    if cls == 0:
        # Normal: periodic QRS complex
        beat = 0.8 * np.exp(-((t % 1.0 - 0.3) ** 2) / 0.002)
        beat += 0.2 * np.exp(-((t % 1.0 - 0.5) ** 2) / 0.005)  # T wave
        beat += 0.1 * np.sin(2 * np.pi * t)  # baseline drift
    else:
        # Arrhythmia: irregular, extra ectopic beats
        beat = 0.6 * np.exp(-((t % 1.0 - 0.3) ** 2) / 0.002)
        beat += 0.4 * np.exp(-((t % 1.0 - 0.15) ** 2) / 0.001)  # ectopic
        beat += 0.3 * np.exp(-((t % 1.0 - 0.7) ** 2) / 0.003)  # extra
        beat += 0.15 * np.sin(2 * np.pi * 1.7 * t)  # irregular baseline
    return beat + noise * np.random.randn(len(t))


def test_ecg():
    print("\n" + "=" * 60)
    print("TEST 1: ECG-like 1D Signals — Normal vs Arrhythmia")
    print("=" * 60)
    np.random.seed(42)
    torch.manual_seed(42)
    n_samples = 2000
    seq_len = 256
    n_classes = 2

    x = np.zeros((n_samples, 1, seq_len), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    t = np.linspace(0, 4, seq_len)
    for i in range(n_samples):
        cls = i % n_classes
        x[i, 0] = generate_ecg_beat(t, cls)
        y[i] = cls

    x_tr, y_tr = torch.from_numpy(x[:1600]), torch.from_numpy(y[:1600])
    x_te, y_te = torch.from_numpy(x[1600:]), torch.from_numpy(y[1600:])
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=64, shuffle=False)

    model = SpectralModel(
        spatial_shape=(seq_len,), in_channels=1, num_classes=n_classes,
        latent_dim=16, num_mixer_layers=3, activation="square",
        mixer_dropout=0.0, classifier_dropout=0.0, norm_type="group",
    )
    print(f"  Params: {count_params(model)}")
    return train_eval(model, train_loader, test_loader, epochs=12, lr=1e-2)


# ---------------------------------------------------------------------------
# Test 2: Audio pitch classification (raw waveforms)
# ---------------------------------------------------------------------------
def generate_tone(freq, duration=1.0, sr=256, noise=0.05):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t)
    return tone + noise * np.random.randn(len(t))


def test_audio_pitch():
    print("\n" + "=" * 60)
    print("TEST 2: Audio Pitch Classification — 5 musical notes (raw wave)")
    print("=" * 60)
    np.random.seed(42)
    torch.manual_seed(42)
    sr = 256
    duration = 1.0
    seq_len = int(sr * duration)
    n_samples = 2500
    n_classes = 5
    # Frequencies: C4=261.6, D4=293.7, E4=329.6, G4=392.0, A4=440.0 Hz (scaled down for sr=256)
    freqs = np.array([5.0, 6.0, 7.0, 8.5, 10.0])  # normalized for sr=256, ~1 sec

    x = np.zeros((n_samples, 1, seq_len), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        cls = i % n_classes
        x[i, 0] = generate_tone(freqs[cls], duration, sr)
        y[i] = cls

    x_tr, y_tr = torch.from_numpy(x[:2000]), torch.from_numpy(y[:2000])
    x_te, y_te = torch.from_numpy(x[2000:]), torch.from_numpy(y[2000:])
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=64, shuffle=False)

    model = SpectralModel(
        spatial_shape=(seq_len,), in_channels=1, num_classes=n_classes,
        latent_dim=16, num_mixer_layers=3, activation="square",
        mixer_dropout=0.0, classifier_dropout=0.0, norm_type="group",
    )
    print(f"  Params: {count_params(model)}")
    return train_eval(model, train_loader, test_loader, epochs=12, lr=1e-2)


# ---------------------------------------------------------------------------
# Test 3: Spectrogram-as-image (2D freq×time patches)
# ---------------------------------------------------------------------------
def make_spectrogram(signal, n_fft=32, hop_length=8):
    """Simple STFT magnitude spectrogram."""
    frames = []
    for i in range(0, len(signal) - n_fft, hop_length):
        frame = signal[i:i + n_fft]
        fft = np.fft.rfft(frame)
        frames.append(np.abs(fft))
    spec = np.stack(frames, axis=1)  # (freq_bins, time_frames)
    return spec.astype(np.float32)


def test_spectrogram():
    print("\n" + "=" * 60)
    print("TEST 3: Spectrogram Classification — 2D freq×time, 4 instrument types")
    print("=" * 60)
    np.random.seed(42)
    torch.manual_seed(42)
    sr = 256
    duration = 1.0
    n_samples = 1200
    n_classes = 4
    freqs = np.array([4.0, 6.0, 8.0, 10.0])
    # Different timbre envelopes per class
    envelopes = [
        lambda t: np.exp(-3 * t),         # pluck
        lambda t: np.ones_like(t),        # organ
        lambda t: np.exp(-0.5 * t) * (1 + 0.3 * np.sin(20 * t)),  # brass
        lambda t: np.exp(-t) * (1 + 0.5 * np.random.randn(len(t))),  # noise
    ]

    specs = []
    labels = []
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    for i in range(n_samples):
        cls = i % n_classes
        tone = np.sin(2 * np.pi * freqs[cls] * t) * envelopes[cls](t)
        tone += 0.05 * np.random.randn(len(t))
        spec = make_spectrogram(tone)
        specs.append(spec[np.newaxis, ...])  # (1, freq, time)
        labels.append(cls)

    x = np.stack(specs, axis=0)  # (N, 1, freq, time)
    y = np.array(labels, dtype=np.int64)
    x_tr, y_tr = torch.from_numpy(x[:960]), torch.from_numpy(y[:960])
    x_te, y_te = torch.from_numpy(x[960:]), torch.from_numpy(y[960:])
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=32, shuffle=False)

    freq_bins, time_frames = x.shape[2], x.shape[3]
    model = SpectralModel(
        spatial_shape=(freq_bins, time_frames), in_channels=1, num_classes=n_classes,
        latent_dim=8, num_mixer_layers=2, activation="square",
        mixer_dropout=0.0, classifier_dropout=0.0, norm_type="batch",
    )
    print(f"  Params: {count_params(model)} | Spectrogram shape: ({freq_bins}, {time_frames})")
    return train_eval(model, train_loader, test_loader, epochs=12, lr=1e-2)


def main():
    device = get_device()
    print(f"Using device: {device}")
    results = {}
    results["ecg_1d"] = test_ecg()
    results["audio_pitch"] = test_audio_pitch()
    results["spectrogram_2d"] = test_spectrogram()

    print("\n" + "=" * 60)
    print("ADVANCED TEST SUMMARY")
    print("=" * 60)
    for name, acc in results.items():
        print(f"  {name:20s}: {acc:.4f}")


if __name__ == "__main__":
    main()
