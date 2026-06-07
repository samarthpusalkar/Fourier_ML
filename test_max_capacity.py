"""
test_max_capacity.py — Push spectral architectures to max on Fashion-MNIST
=========================================================================
Deep (4-8 mixers), wide (latent=32-64), longer epochs (30).
Models tested:
  1. SpectralModel (standard scalar head) — our best so far
  2. SpectralFourierNN (spectral backbone + Fourier layers)
  3. CNN baseline (same depth reference)
"""
import torch
import torch.nn as nn
import numpy as np
import os, gzip, urllib.request
from spectral_core import SpectralModel, count_params
from spectral_fnn import FourierLinear
from test_spectral_fourier_nn import SpectralFourierNN, get_device

FASHION_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_FILES = {
    "train_images": "train-images-idx3-ubyte.gz", "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",   "test_labels": "t10k-labels-idx1-ubyte.gz",
}


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


def train_eval(model, tr, te, epochs=30, lr=1e-2, wd=1e-4, label=""):
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
        if epoch <= 5 or epoch % 5 == 0 or epoch == epochs:
            print(f"  [{label}] Ep{epoch:02d} acc={acc:.4f} (best={best:.4f})")
    print(f"  [{label}] FINAL: {best:.4f}")
    return best


def main():
    dev = get_device()
    print(f"Device: {dev}")
    _download_fashion()
    x_tr, y_tr = _load_fashion("train")
    x_te, y_te = _load_fashion("test")
    tr = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, y_tr), 256, shuffle=True)
    te = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_te, y_te), 256, shuffle=False)

    results = {}

    # -----------------------------------------------------------------------
    # Config A: SpectralModel shallow baseline (12-epoch re-run for control)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\nA: SpectralModel shallow [2 mixers, latent=16, 12ep]\n{'='*60}")
    m = SpectralModel((28,28), 1, 10, latent_dim=16, num_mixer_layers=2,
                      activation="square", mixer_dropout=0.0, classifier_dropout=0.0, norm_type="batch")
    print(f"  Params: {count_params(m):,}")
    results["A_shallow12"] = train_eval(m, tr, te, 12, label="A")

    # -----------------------------------------------------------------------
    # Config B: SpectralModel deep [4 mixers, latent=32, 30ep]
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\nB: SpectralModel deep [4 mixers, latent=32, 30ep]\n{'='*60}")
    m = SpectralModel((28,28), 1, 10, latent_dim=32, num_mixer_layers=4,
                      activation="square", mixer_dropout=0.1, classifier_dropout=0.2, norm_type="batch")
    print(f"  Params: {count_params(m):,}")
    results["B_deep30"] = train_eval(m, tr, te, 30, label="B")

    # -----------------------------------------------------------------------
    # Config C: SpectralModel very deep [6 mixers, latent=48, 30ep]
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\nC: SpectralModel very deep [6 mixers, latent=48, 30ep]\n{'='*60}")
    m = SpectralModel((28,28), 1, 10, latent_dim=48, num_mixer_layers=6,
                      activation="square", mixer_dropout=0.15, classifier_dropout=0.3, norm_type="batch")
    print(f"  Params: {count_params(m):,}")
    results["C_vdeep30"] = train_eval(m, tr, te, 30, label="C")

    # -----------------------------------------------------------------------
    # Config D: SpectralFourierNN deep [4 mixers, latent=32, head=[128,64], 30ep]
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\nD: SpectralFourierNN deep [4 mixers, latent=32, head=[128,64], 30ep]\n{'='*60}")
    m = SpectralFourierNN(
        spatial_shape=(28,28), in_channels=1, num_classes=10,
        latent_dim=32, num_mixer_layers=4, fourier_dims=[128, 64],
        num_modes=32, activation="square", dropout=0.2, norm_type="batch")
    print(f"  Params: {count_params(m):,}")
    results["D_fnn30"] = train_eval(m, tr, te, 30, label="D")

    # -----------------------------------------------------------------------
    # Config E: SpectralModel + wider classifier [4 mixers, latent=32, 256 hidden, 30ep]
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\nE: SpectralModel wide [4 mixers, latent=32, head_hidden=256, 30ep]\n{'='*60}")
    m = SpectralModel((28,28), 1, 10, latent_dim=32, num_mixer_layers=4, head_hidden=256,
                      activation="square", mixer_dropout=0.1, classifier_dropout=0.3, norm_type="batch")
    print(f"  Params: {count_params(m):,}")
    results["E_wide30"] = train_eval(m, tr, te, 30, label="E")

    print(f"\n{'='*60}\nMAX CAPACITY SUMMARY\n{'='*60}")
    for k, v in results.items():
        print(f"  {k:20s}: {v:.4f}")


if __name__ == "__main__":
    main()
