"""
test_spectral_fourier_nn.py — Spectral backbone + Fourier-NN classifier
=======================================================================
Stages:
  1. N-D spectral mixers (native FFT over spatial dims)  <- keeps structure
  2. Global spatial pool  -> (B, latent_dim)
  3. Fourier-NN layers as classifier head

Compare vs standard SpectralModel (spectral mixers + Linear head) on Fashion.
"""
import torch
import torch.nn as nn
import numpy as np
import os, gzip, urllib.request
from spectral_core import ChannelProjection, SpectralMixer, count_params
from spectral_fnn import FourierLinear, make_nufft_grid

FASHION_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_FILES = {
    "train_images": "train-images-idx3-ubyte.gz", "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",   "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def _download_fashion():
    os.makedirs("./data/fashion", exist_ok=True)
    for key, fname in FASHION_FILES.items():
        fpath = os.path.join("./data/fashion", fname)
        if not os.path.exists(fpath):
            urllib.request.urlretrieve(FASHION_URL + fname, fpath)

def _load_fashion(subset="train"):
    prefix = "train" if subset == "train" else "t10k"
    ip = os.path.join("./data/fashion", f"{prefix}-images-idx3-ubyte.gz")
    lp = os.path.join("./data/fashion", f"{prefix}-labels-idx1-ubyte.gz")
    with gzip.open(ip, "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32) / 255.0
    with gzip.open(lp, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    images = np.expand_dims(images, axis=1)
    return torch.from_numpy(images), torch.from_numpy(labels)


# ---------------------------------------------------------------------------
# Spectral backbone → Fourier-NN classifier
# ---------------------------------------------------------------------------
class SpectralFourierNN(nn.Module):
    def __init__(self, spatial_shape, in_channels, num_classes,
                 latent_dim=16, num_mixer_layers=2, fourier_dims=[64],
                 num_modes=16, activation="square", dropout=0.0, norm_type="batch"):
        super().__init__()
        self.channel_proj = ChannelProjection(in_channels, latent_dim, len(spatial_shape))
        self.mixers = nn.ModuleList([
            SpectralMixer(latent_dim, spatial_shape, activation, dropout, norm_type)
            for _ in range(num_mixer_layers)
        ])
        dims = [latent_dim] + fourier_dims + [num_classes]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(FourierLinear(dims[i], dims[i+1], num_modes))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        z = self.channel_proj(x)
        for mixer in self.mixers:
            z = mixer(z)
        z = z.mean(dim=list(range(2, z.ndim)))
        return self.classifier(z)


# ---------------------------------------------------------------------------
# Baseline: SpectralModel (current core)
# ---------------------------------------------------------------------------
from spectral_core import SpectralModel


def train_eval(model, tr, te, epochs=12, lr=1e-2, wd=1e-4):
    dev = get_device()
    model = model.to(dev)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        c, t = 0, 0
        with torch.no_grad():
            for xb, yb in te:
                xb, yb = xb.to(dev), yb.to(dev)
                c += (model(xb).argmax(-1) == yb).sum().item()
                t += yb.size(0)
        acc = c / t; best = max(best, acc)
        if epoch <= 3 or epoch % 3 == 0 or epoch == epochs:
            print(f"  Ep{epoch:02d} acc={acc:.4f}")
    print(f"  Best: {best:.4f}")
    return best


def main():
    dev = get_device()
    print(f"Device: {dev}")
    print("=" * 60)
    print("Spectral backbone + Fourier-NN classifier  (non-flattened)")
    print("=" * 60)
    _download_fashion()
    x_tr, y_tr = _load_fashion("train")
    x_te, y_te = _load_fashion("test")
    tr = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, y_tr), 256, shuffle=True)
    te = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_te, y_te), 256, shuffle=False)

    # --- Fourier-NN head ---
    print("\n--- SpectralFourierNN [2 mixers, head=64, modes=16] ---")
    model1 = SpectralFourierNN(
        spatial_shape=(28, 28), in_channels=1, num_classes=10,
        latent_dim=16, num_mixer_layers=2, fourier_dims=[64],
        num_modes=16, activation="square", dropout=0.0, norm_type="batch",
    )
    print(f"  Params: {count_params(model1):,}")
    acc1 = train_eval(model1, tr, te, 12)

    # --- Standard core ---
    print("\n--- SpectralModel (standard head) ---")
    model2 = SpectralModel(
        spatial_shape=(28, 28), in_channels=1, num_classes=10,
        latent_dim=16, num_mixer_layers=2, activation="square",
        mixer_dropout=0.0, classifier_dropout=0.0, norm_type="batch",
    )
    print(f"  Params: {count_params(model2):,}")
    acc2 = train_eval(model2, tr, te, 12)

    # --- CNN baseline ---
    print("\n--- CNN Baseline ---")
    cnn = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Linear(128, 10)
    )
    print(f"  Params: {count_params(cnn):,}")
    acc3 = train_eval(cnn, tr, te, 12)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  SpectralFourierNN: {acc1:.4f}")
    print(f"  SpectralModel:     {acc2:.4f}")
    print(f"  CNN Baseline:      {acc3:.4f}")


if __name__ == "__main__":
    main()
