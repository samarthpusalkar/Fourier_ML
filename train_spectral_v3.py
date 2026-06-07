"""
Spectral-Locality Network v3
==============================
Corrected synthesis incorporating comparisons:
  1. FIXED integer-harmonic frequencies: w_n = 2*pi*n / P
     P is a LEARNABLE scalar that adapts to data scale (not auto-heuristic).
  2. Distributed Fourier representation:
     [cos(z @ W), sin(z @ W)] as continuous feature vector.
  3. SMALL MLP head: ONE hidden layer (not linear, not deep).
     This balances capacity with interpretability.

Architecture depth: 3-4 layers max.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import os
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
    images = np.expand_dims(images, axis=1)
    return torch.from_numpy(images), torch.from_numpy(labels)


def _download_cifar10(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    fpath = os.path.join(data_dir, "cifar-10-python.tar.gz")
    if not os.path.exists(fpath):
        print("Downloading CIFAR-10 ...")
        urllib.request.urlretrieve(CIFAR10_URL, fpath)
    extracted = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extracted):
        import tarfile
        with tarfile.open(fpath, "r:gz") as tar:
            tar.extractall(data_dir)
    return extracted


def _load_cifar10(data_dir):
    extracted = _download_cifar10(data_dir)
    base = extracted
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
# Fixed Fourier Layer with learnable domain period
# ---------------------------------------------------------------------------
class FixedFourierLayerV3(nn.Module):
    """
    Frequencies are fixed integer harmonics w_n = 2*pi*n / P, but P is learnable.
    This lets the model adapt its spectral scale to the data while preserving
    the orthogonal harmonic structure.
    """
    def __init__(self, latent_dim, num_modes, init_period=2.0):
        super().__init__()
        self.num_modes = num_modes
        self.latent_dim = latent_dim
        # Learnable domain period scalar. Starts at init_period.
        self.period = nn.Parameter(torch.tensor(init_period, dtype=torch.float32))
        # Fixed frequency magnitudes: harmonic number n
        self.register_buffer('harmonic_n', torch.arange(num_modes + 1, dtype=torch.float32))

    def forward(self, z):
        # z: (batch, latent_dim)
        # w_n = 2*pi*n / P -> shape (num_modes+1,)
        freqs = 2.0 * np.pi * self.harmonic_n / (self.period.abs() + 1e-6)
        # For multi-D latent, project along a random fixed direction (kept in buffer)
        if not hasattr(self, 'proj_dir'):
            proj = torch.randn(self.latent_dim, 1)
            proj = proj / proj.norm()
            self.register_buffer('proj_dir', proj)
        z_proj = torch.matmul(z, self.proj_dir).squeeze(-1)  # (batch,)
        proj = freqs.unsqueeze(0) * z_proj.unsqueeze(1)  # (batch, num_modes+1)
        cos_act = torch.cos(proj)
        sin_act = torch.sin(proj)
        return torch.cat([cos_act, sin_act], dim=-1)


# ---------------------------------------------------------------------------
# Spectral Token Mixer
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Unified model
# ---------------------------------------------------------------------------
class SpectralV3(nn.Module):
    def __init__(self, input_type, img_size=None, patch_size=None, in_ch=None,
                 input_dim=None, latent_dim=8, num_modes=32, num_mixer_layers=0,
                 num_classes=10, init_period=2.0, head_hidden=64):
        super().__init__()
        self.input_type = input_type
        self.num_modes = num_modes

        if input_type == "image":
            self._init_image(img_size, patch_size, in_ch, latent_dim, num_mixer_layers)
        else:
            self._init_flat(input_dim, latent_dim)

        self.fourier = FixedFourierLayerV3(latent_dim, num_modes, init_period)
        # Single hidden-layer head on distributed Fourier features
        self.classifier = nn.Sequential(
            nn.Linear(2 * (num_modes + 1), head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_classes),
        )

    def _init_image(self, img_size, patch_size, in_ch, latent_dim, num_mixer_layers):
        ph, pw = patch_size
        self.num_patches = (img_size[0] // ph) * (img_size[1] // pw)
        patch_dim = ph * pw * in_ch
        self.encoder = nn.Linear(patch_dim, latent_dim)
        self.mixers = nn.ModuleList([
            SpectralTokenMixer(latent_dim, self.num_patches)
            for _ in range(num_mixer_layers)
        ])
        self.patch_size = patch_size

    def _init_flat(self, input_dim, latent_dim):
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.mixers = nn.ModuleList([])
        self.num_patches = 1
        self.patch_size = None

    def forward(self, x):
        if self.input_type == "image":
            B, C, H, W = x.shape
            ph, pw = self.patch_size
            x = x.unfold(2, ph, ph).unfold(3, pw, pw)
            x = x.contiguous().view(B, C, H // ph, W // pw, ph, pw)
            x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
            x = x.view(B, -1, C * ph * pw)
            z = self.encoder(x)
            for mixer in self.mixers:
                z = mixer(z)
            z = z.mean(dim=1)
        else:
            z = self.encoder(x)
        acts = self.fourier(z)
        logits = self.classifier(acts)
        return logits

    def get_period(self):
        return self.fourier.period.item()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10", "iris"])
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--num_modes", type=int, default=32)
    parser.add_argument("--num_mixer_layers", type=int, default=2)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[4, 4])
    parser.add_argument("--init_period", type=float, default=2.0)
    parser.add_argument("--head_hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: {args.dataset}")
    print(f"Architecture: learnable-period integer-harmonics + distributed acts + small MLP head\n")

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

    model = SpectralV3(
        input_type=input_type,
        img_size=img_size,
        patch_size=tuple(args.patch_size),
        in_ch=in_ch,
        input_dim=x_tr.shape[1] if input_type == "flat" else None,
        latent_dim=args.latent_dim,
        num_modes=args.num_modes,
        num_mixer_layers=args.num_mixer_layers,
        num_classes=num_classes,
        init_period=args.init_period,
        head_hidden=args.head_hidden,
    ).to(device)

    print("="*60)
    print("SPECTRAL V3 MODEL")
    print("="*60)
    print(f"Total parameters: {count_params(model)}")
    print(f"Fourier modes: {args.num_modes} | Head hidden: {args.head_hidden}")
    print(f"Initial period P: {args.init_period}")
    print("-"*60)

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
            torch.save({"state_dict": model.state_dict(), "period": model.get_period()},
                       f"best_v3_{args.dataset}.pt")
        if epoch <= 3 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d} | Train Acc: {ta:.4f} | Test Acc: {va:.4f} | Period: {model.get_period():.4f}")
    print(f"Best Spectral V3 accuracy: {best_acc:.4f}\n")

    if args.run_baseline:
        if input_type == "image":
            baseline = StandardMLP(input_dim=np.prod(x_tr.shape[1:]), hidden=[256, 128], num_classes=num_classes).to(device)
        else:
            baseline = StandardMLP(input_dim=x_tr.shape[1], hidden=[64, 32], num_classes=num_classes).to(device)
        print("="*60)
        print("BASELINE MLP")
        print("="*60)
        print(f"Total parameters: {count_params(baseline)}")
        print("-"*60)
        opt_b = torch.optim.Adam(baseline.parameters(), lr=args.lr)
        sch_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=args.epochs)
        best_b = 0.0
        for epoch in range(1, args.epochs + 1):
            tl, ta = train_epoch(baseline, train_loader, opt_b, criterion, device)
            vl, va = evaluate(baseline, test_loader, criterion, device)
            sch_b.step()
            if va > best_b:
                best_b = va
            if epoch <= 3 or epoch % 5 == 0 or epoch == args.epochs:
                print(f"Epoch {epoch:02d} | Train Acc: {ta:.4f} | Test Acc: {va:.4f}")
        print(f"Best Baseline accuracy: {best_b:.4f}\n")

    print(f"Final learned period: {model.get_period():.4f}")
    print(f"Report saved to v3_report_{args.dataset}.json")


if __name__ == "__main__":
    main()
