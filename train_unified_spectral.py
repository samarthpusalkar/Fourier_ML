"""
Unified Spectral-Locality Trainer
===================================
Runs the spectral N-D model (or flat variant) on:
  - MNIST   (1x28x28, 10 classes)
  - CIFAR-10 (3x32x32, 10 classes)
  - Iris     (4 features, 3 classes, tabular)

Architecture auto-adapts:
  - Images: PatchEmbed -> PatchLatentEncoder -> SpectralTokenMixer -> FourierHead
  - Tabular: Flat LatentEncoder -> FourierHead (no mixer needed for tiny input)

Also supports LR sweep for spectral parameter sensitivity testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Data loaders (no torchvision dependency for images)
# ---------------------------------------------------------------------------
import gzip
import urllib.request
import pickle

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _download_mnist(data_dir):
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    os.makedirs(data_dir, exist_ok=True)
    for key, fname in files.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(MNIST_URL + fname, fpath)


def _load_mnist(data_dir, subset="train"):
    prefix = "train" if subset == "train" else "t10k"
    img_path = os.path.join(data_dir, f"{prefix}-images-idx3-ubyte.gz")
    lbl_path = os.path.join(data_dir, f"{prefix}-labels-idx1-ubyte.gz")
    with gzip.open(img_path, "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    images = images.reshape(-1, 28, 28).astype(np.float32) / 255.0
    with gzip.open(lbl_path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    labels = labels.astype(np.int64)
    # Add channel dim: (N, 1, 28, 28)
    images = np.expand_dims(images, axis=1)
    return torch.from_numpy(images), torch.from_numpy(labels)


def _download_cifar10(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    fpath = os.path.join(data_dir, "cifar-10-python.tar.gz")
    if not os.path.exists(fpath):
        print("Downloading CIFAR-10 ...")
        urllib.request.urlretrieve(CIFAR10_URL, fpath)
    # Extract if needed
    extracted = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extracted):
        import tarfile
        with tarfile.open(fpath, "r:gz") as tar:
            tar.extractall(data_dir)
    return extracted


def _load_cifar10(data_dir):
    extracted = _download_cifar10(data_dir)
    base = extracted

    # Training batches
    x_train = []
    y_train = []
    for i in range(1, 6):
        batch_path = os.path.join(base, f"data_batch_{i}")
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        x_train.append(batch[b"data"])
        y_train.extend(batch[b"labels"])
    x_train = np.concatenate(x_train, axis=0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_train = np.array(y_train, dtype=np.int64)

    # Test batch
    test_path = os.path.join(base, "test_batch")
    with open(test_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    x_test = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_test = np.array(batch[b"labels"], dtype=np.int64)

    return torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(x_test), torch.from_numpy(y_test)


def _load_iris():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    X, y = load_iris(return_X_y=True)
    X = X.astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return (
        torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64)),
        torch.from_numpy(x_test), torch.from_numpy(y_test.astype(np.int64))
    )


# ---------------------------------------------------------------------------
# Model components (same as train_spectral_nd.py but generalized)
# ---------------------------------------------------------------------------
class PatchEmbed1D(nn.Module):
    def __init__(self, img_size, patch_size, in_ch):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_dim = patch_size[0] * patch_size[1] * in_ch

    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)
        x = x.contiguous().view(B, C, H // ph, W // pw, ph, pw)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * ph * pw)
        return x


class PatchLatentEncoder(nn.Module):
    def __init__(self, patch_dim, latent_dim, hidden_dims):
        super().__init__()
        layers = []
        prev = patch_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        B, N, D = x.shape
        z = self.net(x.view(B * N, D))
        return z.view(B, N, -1)


class SpectralTokenMixer(nn.Module):
    def __init__(self, latent_dim, num_patches):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_patches = num_patches
        n_rfft = num_patches // 2 + 1
        self.gain_real = nn.Parameter(torch.ones(latent_dim, n_rfft) * 0.5)
        self.gain_imag = nn.Parameter(torch.zeros(latent_dim, n_rfft))
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z):
        z_t = z.transpose(1, 2)
        z_fft = torch.fft.rfft(z_t, dim=-1)
        gain = torch.view_as_complex(torch.stack([self.gain_real, self.gain_imag], dim=-1))
        z_filtered = z_fft * gain.unsqueeze(0)
        z_out = torch.fft.irfft(z_filtered, n=self.num_patches, dim=-1)
        z_out = z_out.transpose(1, 2)
        return self.norm(z + z_out)


class FourierHead(nn.Module):
    def __init__(self, latent_dim, num_modes, num_classes):
        super().__init__()
        self.num_modes = num_modes
        self.frequencies = nn.Parameter(torch.randn(num_modes, latent_dim) * 0.1)
        self.A = nn.Parameter(torch.randn(num_modes) * 0.01)
        self.B = nn.Parameter(torch.randn(num_modes) * 0.01)
        self.dc = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Sequential(
            nn.Linear(1 + latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, z_pooled):
        proj = torch.matmul(z_pooled, self.frequencies.T)
        fourier_scalar = self.dc + torch.sum(
            self.A * torch.cos(proj) + self.B * torch.sin(proj), dim=-1, keepdim=True
        )
        features = torch.cat([fourier_scalar, z_pooled], dim=-1)
        return self.classifier(features)

    def get_spectral_params(self):
        return {
            "frequencies": self.frequencies.detach().cpu().numpy(),
            "A": self.A.detach().cpu().numpy(),
            "B": self.B.detach().cpu().numpy(),
            "dc": self.dc.detach().cpu().item(),
        }


class FlatLatentEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UnifiedSpectralModel(nn.Module):
    """
    Auto-switch between image (patch) and flat (tabular) modes.
    """
    def __init__(self, input_type="image", img_size=(28,28), patch_size=(4,4), in_ch=1,
                 input_dim=None, latent_dim=8, patch_hidden=[32], flat_hidden=[64],
                 num_mixer_layers=2, num_modes=64, num_classes=10):
        super().__init__()
        self.input_type = input_type

        if input_type == "image":
            self.patch_embed = PatchEmbed1D(img_size, patch_size, in_ch)
            num_patches = self.patch_embed.num_patches
            patch_dim = self.patch_embed.patch_dim
            self.encoder = PatchLatentEncoder(patch_dim, latent_dim, patch_hidden)
            self.mixers = nn.ModuleList([
                SpectralTokenMixer(latent_dim, num_patches)
                for _ in range(num_mixer_layers)
            ])
        else:
            self.patch_embed = None
            self.encoder = FlatLatentEncoder(input_dim, latent_dim, flat_hidden)
            self.mixers = nn.ModuleList([])  # no token mixing needed for tiny tabular

        self.head = FourierHead(latent_dim, num_modes, num_classes)

    def forward(self, x):
        if self.input_type == "image":
            patches = self.patch_embed(x)
            z = self.encoder(patches)
            for mixer in self.mixers:
                z = mixer(z)
            z_pooled = z.mean(dim=1)
        else:
            z_pooled = self.encoder(x)
        return self.head(z_pooled)

    def get_fourier_params(self):
        return self.head.get_spectral_params()


# ---------------------------------------------------------------------------
# Standard baselines for comparison
# ---------------------------------------------------------------------------
class StandardMLP(nn.Module):
    def __init__(self, input_dim, hidden, num_classes):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return total_loss / len(loader), correct / total


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10", "iris"])
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--num_modes", type=int, default=64)
    parser.add_argument("--num_mixer_layers", type=int, default=2)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[4, 4])
    parser.add_argument("--flat_hidden", type=int, nargs="+", default=[64])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: {args.dataset}\n")

    # --- Load data ---
    if args.dataset == "mnist":
        _download_mnist(os.path.join(args.data_dir, "mnist"))
        x_tr, y_tr = _load_mnist(os.path.join(args.data_dir, "mnist"), "train")
        x_te, y_te = _load_mnist(os.path.join(args.data_dir, "mnist"), "test")
        input_type = "image"
        img_size = (28, 28)
        in_ch = 1
        num_classes = 10
    elif args.dataset == "cifar10":
        x_tr, y_tr, x_te, y_te = _load_cifar10(os.path.join(args.data_dir, "cifar10"))
        input_type = "image"
        img_size = (32, 32)
        in_ch = 3
        num_classes = 10
    elif args.dataset == "iris":
        x_tr, y_tr, x_te, y_te = _load_iris()
        input_type = "flat"
        img_size = None
        in_ch = None
        num_classes = 3

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=args.batch_size, shuffle=False
    )

    # --- Spectral model ---
    model = UnifiedSpectralModel(
        input_type=input_type,
        img_size=img_size,
        patch_size=tuple(args.patch_size),
        in_ch=in_ch,
        input_dim=x_tr.shape[1] if input_type == "flat" else None,
        latent_dim=args.latent_dim,
        patch_hidden=[32],
        flat_hidden=args.flat_hidden,
        num_mixer_layers=args.num_mixer_layers,
        num_modes=args.num_modes,
        num_classes=num_classes,
    ).to(device)

    print("=" * 60)
    print("SPECTRAL MODEL")
    print("=" * 60)
    print(f"Total parameters: {count_params(model)}")
    print("-" * 60)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        if va > best_acc:
            best_acc = va
            ckpt = {"state_dict": model.state_dict(), "fourier_params": model.get_fourier_params()}
            torch.save(ckpt, f"best_{args.dataset}_spectral.pt")
        print(f"Epoch {epoch:02d} | Train Acc: {ta:.4f} | Test Acc: {va:.4f}")
    print(f"Best Spectral accuracy: {best_acc:.4f}\n")

    # --- Optional baseline ---
    if args.run_baseline:
        if input_type == "image":
            baseline = StandardMLP(input_dim=np.prod(x_tr.shape[1:]), hidden=[256, 128], num_classes=num_classes).to(device)
        else:
            baseline = StandardMLP(input_dim=x_tr.shape[1], hidden=[64, 32], num_classes=num_classes).to(device)
        print("=" * 60)
        print("BASELINE MLP")
        print("=" * 60)
        print(f"Total parameters: {count_params(baseline)}")
        print("-" * 60)
        opt_b = torch.optim.Adam(baseline.parameters(), lr=args.lr)
        sch_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=args.epochs)
        best_b = 0.0
        for epoch in range(1, args.epochs + 1):
            tl, ta = train_epoch(baseline, train_loader, opt_b, criterion, device)
            vl, va = evaluate(baseline, test_loader, criterion, device)
            sch_b.step()
            if va > best_b:
                best_b = va
            print(f"Epoch {epoch:02d} | Train Acc: {ta:.4f} | Test Acc: {va:.4f}")
        print(f"Best Baseline accuracy: {best_b:.4f}\n")

    # --- Print learned Fourier params ---
    fp = model.get_fourier_params()
    freqs = fp["frequencies"]
    A = fp["A"]
    B = fp["B"]
    energy = np.sqrt(A**2 + B**2)
    idx = np.argsort(-energy)
    print("=" * 60)
    print("Top Fourier Head Modes")
    print("=" * 60)
    for rank, i in enumerate(idx[:8], 1):
        print(f"Rank {rank} | Mode {i:2d} | freq_norm={np.linalg.norm(freqs[i]):.3f} | energy={energy[i]:.3f}")
    print(f"DC = {fp['dc']:.4f}")


if __name__ == "__main__":
    main()
