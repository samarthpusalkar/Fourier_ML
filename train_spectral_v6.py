"""
Spectral v6: Global 2D FFT + Learnable Square Wave
===================================================

No rigid patches. Global 2D FFT over full image per channel.
1x1 pointwise conv for channel mixing (not spatial).
Learnable square wave activation between mixer layers:
    y = A * tanh(k * sin(ω * x + φ))

PCA overview:
- Standard PCA: SVD on covariance. O(d^3) for d dimensions. Linear only.
- Limitations: destroys spatial structure (flattened covariance),
  not shift/scale/rotation invariant, not differentiable naturally.
- Works when data near low-dim linear subspace (eigenfaces).
- Fails on spatial data because local correlations become
global entanglement after flattening.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import gzip
import urllib.request
import pickle

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
FASHION_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


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


def _download_fashion(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for key, fname in FASHION_FILES.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            urllib.request.urlretrieve(FASHION_URL + fname, fpath)


def _load_fashion(data_dir, subset="train"):
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
    x_train, y_train = [], []
    for i in range(1, 6):
        with open(os.path.join(base, f"data_batch_{i}"), "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        x_train.append(batch[b"data"])
        y_train.extend(batch[b"labels"])
    x_train = np.concatenate(x_train, axis=0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_train = np.array(y_train, dtype=np.int64)
    with open(os.path.join(base, "test_batch"), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    x_test = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_test = np.array(batch[b"labels"], dtype=np.int64)
    return torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(x_test), torch.from_numpy(y_test)


# ---------------------------------------------------------------------------
# Learnable Square Wave Activation
# ---------------------------------------------------------------------------
class LearnableSquareWave(nn.Module):
    """
    y = A * tanh(k * sin(omega * x + phi))
    A: amplitude, learnable scalar
    omega: frequency, learnable scalar (exp(log_freq))
    phi: phase, learnable scalar
    k: steepness, fixed (controls squareness)
    """
    def __init__(self, steepness=10.0):
        super().__init__()
        self.amplitude = nn.Parameter(torch.tensor(1.0))
        self.log_freq = nn.Parameter(torch.tensor(0.0))
        self.phase = nn.Parameter(torch.tensor(0.0))
        self.steepness = steepness

    def forward(self, x):
        freq = torch.exp(self.log_freq)
        return self.amplitude * torch.tanh(self.steepness * torch.sin(freq * x + self.phase))


# ---------------------------------------------------------------------------
# Chebyshev Polynomial Activation
# ---------------------------------------------------------------------------
class ChebyshevActivation(nn.Module):
    """
    Weighted sum of Chebyshev polynomials T_n(x).
    T_n(x) = cos(n * arccos(x))  for x in [-1,1].
    Spectral-native: T_n(cos theta) = cos(n*theta) is exactly a Fourier term.
    Bounded, orthogonal, no sharp corners.
    """
    def __init__(self, order=4):
        super().__init__()
        self.order = order
        self.weights = nn.Parameter(torch.randn(order + 1) * 0.1)

    def forward(self, x):
        # Clamp to [-1,1] for Chebyshev domain
        x = torch.tanh(x)
        # Recurrence: T_0=1, T_1=x, T_{n+1}=2xT_n - T_{n-1}
        T = [torch.ones_like(x), x]
        for n in range(2, self.order + 1):
            T.append(2 * x * T[-1] - T[-2])
        out = sum(self.weights[n] * T[n] for n in range(self.order + 1))
        return out


# ---------------------------------------------------------------------------
# Frequency Modulation (FM) Activation
# ---------------------------------------------------------------------------
class FMActivation(nn.Module):
    """
    y = tanh(alpha * sin(w1*x + A * sin(w2*x)))
    Carrier (w1), modulator (w2), depth (A), gain (alpha).
    Creates sideband harmonics. Rich spectral structure.
    """
    def __init__(self):
        super().__init__()
        self.log_w1 = nn.Parameter(torch.tensor(0.0))
        self.log_w2 = nn.Parameter(torch.tensor(0.0))
        self.log_A = nn.Parameter(torch.tensor(0.0))
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        w1 = torch.exp(self.log_w1)
        w2 = torch.exp(self.log_w2)
        A = torch.exp(self.log_A)
        alpha = torch.exp(self.log_alpha)
        mod = w1 * x + A * torch.sin(w2 * x)
        return torch.tanh(alpha * torch.sin(mod))


# ---------------------------------------------------------------------------
# Global Spectral Mixer: 2D FFT over full image, no patches
# ---------------------------------------------------------------------------
class GlobalSpectralMixer(nn.Module):
    """
    2D FFT on each channel over full HxW image.
    Learnable complex gains per frequency bin.
    Activation + norm + residual.
    """
    def __init__(self, channels, img_size, activation="relu"):
        super().__init__()
        self.channels = channels
        h, w = img_size
        self.freq_shape = (h, w // 2 + 1)
        self.gain_real = nn.Parameter(torch.ones(channels, *self.freq_shape) * 0.5)
        self.gain_imag = nn.Parameter(torch.zeros(channels, *self.freq_shape))
        self.norm = nn.GroupNorm(num_groups=max(1, channels // 4), num_channels=channels)

        act_map = {
            "none": nn.Identity(),
            "relu": nn.ReLU(),
            "square": LearnableSquareWave(steepness=10.0),
            "gelu": nn.GELU(),
            "chebyshev": ChebyshevActivation(order=4),
            "fm": FMActivation(),
        }
        self.activation = act_map.get(activation, nn.Identity())

    def forward(self, x):
        x_fft = torch.fft.rfft2(x, dim=(-2, -1))
        gain = torch.view_as_complex(torch.stack([self.gain_real, self.gain_imag], dim=-1))
        x_filtered = x_fft * gain.unsqueeze(0)
        x_out = torch.fft.irfft2(x_filtered, s=x.shape[-2:], dim=(-2, -1))
        x_out = self.activation(x_out)
        return self.norm(x_out + x)


# ---------------------------------------------------------------------------
# Non-uniform frequency step: dense near 0, uniform beyond cap
# ---------------------------------------------------------------------------
def make_freq_grid(num_modes, mode="uniform"):
    """
    mode="uniform":  0, 1, 2, ... N  (constant step)
    mode="dense_low": k indices densely packed in [0, 1], rest uniform in [1, N]
      e.g. N=64: first 32 modes at 0, 1/32, 2/32, ..., 31/32
                  then 32, 33, ..., 64  (step=1)
    """
    if mode == "uniform":
        return torch.arange(num_modes + 1, dtype=torch.float32)
    elif mode == "dense_low":
        half = num_modes // 2
        dense = torch.linspace(0.0, 1.0, steps=half + 1)[:-1]  # 0, 1/32, ..., 31/32
        rest = torch.arange(1, half + 2, dtype=torch.float32)  # 1, 2, ..., half+1
        return torch.cat([dense, rest])
    elif mode == "split":
        # 60% modes densely in [0, 1), 40% sparsely in [1, num_modes]
        n_dense = int((num_modes + 1) * 0.6)
        n_rest = (num_modes + 1) - n_dense
        dense = torch.linspace(0.0, 1.0, steps=n_dense + 1)[:-1]  # 0, 1/n, 2/n, ..., (n-1)/n
        # rest: uniform from 1 to num_modes, exactly n_rest points
        rest = torch.linspace(1.0, float(num_modes), steps=n_rest)
        return torch.cat([dense, rest])
    else:
        raise ValueError(mode)


# ---------------------------------------------------------------------------
# Fixed-integer-harmonic Fourier head with non-uniform step option
# ---------------------------------------------------------------------------
class GenericFourierHead(nn.Module):
    def __init__(self, latent_dim, num_modes, init_scale=2.0, spacing_mode="uniform"):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.spacing_mode = spacing_mode
        self.proj_weight = nn.Parameter(torch.randn(latent_dim) * 0.1)
        self.proj_bias = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        grid = make_freq_grid(num_modes, spacing_mode)
        self.register_buffer('harmonic_n', grid)

    def forward(self, z):
        z_1d = torch.matmul(z, self.proj_weight) + self.proj_bias
        freqs = 2.0 * np.pi * self.harmonic_n / (self.scale.abs() + 1e-6)
        proj = z_1d.unsqueeze(1) * freqs.unsqueeze(0)
        cos_terms = torch.cos(proj)
        sin_terms = torch.sin(proj)
        fourier_scalar = cos_terms.sum(dim=-1, keepdim=True) + sin_terms.sum(dim=-1, keepdim=True)
        fourier_scalar = fourier_scalar / (self.num_modes + 1)
        return fourier_scalar, z


# ---------------------------------------------------------------------------
# Global Spectral Model: no patches, full-image FFT
# ---------------------------------------------------------------------------
class GlobalSpectralModel(nn.Module):
    def __init__(self, img_size, in_ch, latent_dim=16, num_modes=64,
                 num_mixer_layers=2, num_classes=10, head_hidden=128,
                 init_scale=2.0, activation="relu", spacing_mode="uniform"):
        super().__init__()
        self.latent_dim = latent_dim
        self.channel_proj = nn.Conv2d(in_ch, latent_dim, kernel_size=1)

        self.mixers = nn.ModuleList([
            GlobalSpectralMixer(latent_dim, img_size, activation)
            for _ in range(num_mixer_layers)
        ])

        self.fourier_head = GenericFourierHead(latent_dim, num_modes, init_scale, spacing_mode)
        self.classifier = nn.Sequential(
            nn.Linear(1 + latent_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x):
        z = self.channel_proj(x)
        for mixer in self.mixers:
            z = mixer(z)
        z = z.mean(dim=[-2, -1])
        fourier_scalar, z_full = self.fourier_head(z)
        features = torch.cat([fourier_scalar, z_full], dim=-1)
        logits = self.classifier(features)
        return logits

    def get_fourier_info(self):
        return {
            "scale": self.fourier_head.scale.item(),
            "proj_weight_norm": self.fourier_head.proj_weight.norm().item(),
        }


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
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "fashion", "cifar10"])
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--num_modes", type=int, default=64)
    parser.add_argument("--num_mixer_layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="relu", choices=["none", "relu", "square", "gelu", "chebyshev", "fm"])
    parser.add_argument("--spacing", type=str, default="split", choices=["uniform", "dense_low", "split"])
    parser.add_argument("--init_scale", type=float, default=2.0)
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "mnist":
        _download_mnist(os.path.join(args.data_dir, "mnist"))
        x_tr, y_tr = _load_mnist(os.path.join(args.data_dir, "mnist"), "train")
        x_te, y_te = _load_mnist(os.path.join(args.data_dir, "mnist"), "test")
        img_size = (28, 28); in_ch = 1; num_classes = 10
    elif args.dataset == "fashion":
        _download_fashion(os.path.join(args.data_dir, "fashion"))
        x_tr, y_tr = _load_fashion(os.path.join(args.data_dir, "fashion"), "train")
        x_te, y_te = _load_fashion(os.path.join(args.data_dir, "fashion"), "test")
        img_size = (28, 28); in_ch = 1; num_classes = 10
    elif args.dataset == "cifar10":
        x_tr, y_tr, x_te, y_te = _load_cifar10(os.path.join(args.data_dir, "cifar10"))
        img_size = (32, 32); in_ch = 3; num_classes = 10

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=args.batch_size, shuffle=False
    )

    model = GlobalSpectralModel(
        img_size=img_size, in_ch=in_ch, latent_dim=args.latent_dim,
        num_modes=args.num_modes, num_mixer_layers=args.num_mixer_layers,
        num_classes=num_classes, head_hidden=args.head_hidden,
        init_scale=args.init_scale, activation=args.activation,
        spacing_mode=args.spacing,
    ).to(device)

    print(f"Device: {device} | Dataset: {args.dataset}")
    print(f"Architecture: Global 2D FFT + 1x1 conv + {args.activation} activation")
    print(f"No rigid patches. Full image frequency filtering.\n")
    print("="*60)
    print("GLOBAL SPECTRAL MODEL")
    print("="*60)
    print(f"Latent dim: {args.latent_dim} | Mixer layers: {args.num_mixer_layers}")
    print(f"Activation: {args.activation} | Total parameters: {count_params(model)}")
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
        if epoch <= 3 or epoch % 5 == 0 or epoch == args.epochs:
            info = model.get_fourier_info()
            print(f"Epoch {epoch:02d} | Train Acc: {ta:.4f} | Test Acc: {va:.4f} | Scale: {info['scale']:.4f}")
    print(f"Best Global Spectral accuracy: {best_acc:.4f}\n")

    if args.run_baseline:
        baseline = StandardMLP(input_dim=np.prod(x_tr.shape[1:]), hidden=[256, 128], num_classes=num_classes).to(device)
        print("="*60); print("BASELINE MLP"); print("="*60)
        print(f"Total parameters: {count_params(baseline)}"); print("-"*60)
        opt_b = torch.optim.Adam(baseline.parameters(), lr=args.lr)
        sch_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=args.epochs)
        best_b = 0.0
        for epoch in range(1, args.epochs + 1):
            tl, ta = train_epoch(baseline, train_loader, opt_b, criterion, device)
            vl, va = evaluate(baseline, test_loader, criterion, device)
            sch_b.step()
            if va > best_b: best_b = va
            if epoch <= 3 or epoch % 5 == 0 or epoch == args.epochs:
                print(f"Epoch {epoch:02d} | Train Acc: {ta:.4f} | Test Acc: {va:.4f}")
        print(f"Best Baseline accuracy: {best_b:.4f}\n")


if __name__ == "__main__":
    main()
