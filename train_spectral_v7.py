"""
Spectral v7: Non-Uniform Frequency Spacing + Per-Layer Activation Mixing
========================================================================

Key improvements:
1. Non-uniform frequency spacing (power-law α or log) for higher
   resolution at low frequencies where precision matters.
2. Full Fourier basis: DC + sin + cos (complete, not sine-only).
3. Per-layer activation selection (mix complexity across depth).
4. Learnable frequency/phase in square wave activation per layer.
5. Global 2D FFT, no rigid patches.
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
# Non-Uniform Frequency Spacing
# ---------------------------------------------------------------------------
def make_frequency_spacing(num_modes, mode="uniform", alpha=2.0):
    """
    mode="uniform":  n = 0, 1, 2, 3, ... N
    mode="power":     n = N * (k/N)^alpha  for k=0..N  (clusters near 0 when alpha<1)
    mode="log":       n = N * log(1+c*k)/log(1+c*N)
    """
    n = torch.arange(num_modes + 1, dtype=torch.float32)
    if mode == "uniform":
        return n
    elif mode == "power":
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        return (n / num_modes) ** alpha * num_modes
    elif mode == "log":
        c = 1.0
        return torch.log1p(c * n) / torch.log1p(torch.tensor(c * num_modes)) * num_modes
    else:
        raise ValueError(f"Unknown spacing: {mode}")


# ---------------------------------------------------------------------------
# Complete Fourier Head: DC + Sin + Cos
# ---------------------------------------------------------------------------
class CompleteFourierHead(nn.Module):
    """
    Full Fourier series basis with non-uniform spacing:
    y = a0 + sum_n [an * cos(wn*x) + bn * sin(wn*x)]
    wn = 2*pi * spacing(n) / scale
    """
    def __init__(self, latent_dim, num_modes=64, init_scale=2.0, spacing_mode="uniform", alpha=2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.spacing_mode = spacing_mode
        self.alpha = alpha

        # Projection to scalar
        self.proj_weight = nn.Parameter(torch.randn(latent_dim) * 0.1)
        self.proj_bias = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

        # Fourier coefficients (learnable)
        self.a0 = nn.Parameter(torch.zeros(1))
        self.a_n = nn.Parameter(torch.randn(num_modes) * 0.1)
        self.b_n = nn.Parameter(torch.randn(num_modes) * 0.1)

        spacing = make_frequency_spacing(num_modes, spacing_mode, alpha)
        self.register_buffer('spacing', spacing)

    def forward(self, z):
        z_1d = torch.matmul(z, self.proj_weight) + self.proj_bias
        # Use spacing[1:] for harmonics (skip DC at index 0)
        freqs = 2.0 * np.pi * self.spacing[1:] / (self.scale.abs() + 1e-6)

        proj = z_1d.unsqueeze(1) * freqs.unsqueeze(0)
        cos_terms = self.a_n * torch.cos(proj)
        sin_terms = self.b_n * torch.sin(proj)

        # DC term
        fourier_out = self.a0 + cos_terms.sum(dim=1) + sin_terms.sum(dim=1)

        # Normalization: divide by (num_modes + 1)
        fourier_out = fourier_out / (self.num_modes + 1)
        return fourier_out.unsqueeze(-1), z


# ---------------------------------------------------------------------------
# Per-Layer Activation with learnable square wave per layer
# ---------------------------------------------------------------------------
class LearnableSquareWave(nn.Module):
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
# Global Spectral Mixer
# ---------------------------------------------------------------------------
class GlobalSpectralMixer(nn.Module):
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
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "elu": nn.ELU(),
            "square": LearnableSquareWave(steepness=10.0),
            "softplus": nn.Softplus(),
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
# Spectral v7 Model
# ---------------------------------------------------------------------------
class SpectralV7(nn.Module):
    def __init__(self, img_size, in_ch, latent_dim=16, num_modes=64,
                 num_mixer_layers=2, num_classes=10, head_hidden=128,
                 init_scale=2.0, spacing_mode="uniform", alpha=2.0,
                 activations=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.channel_proj = nn.Conv2d(in_ch, latent_dim, kernel_size=1)

        if activations is None:
            activations = ["relu"] * num_mixer_layers

        self.mixers = nn.ModuleList([
            GlobalSpectralMixer(latent_dim, img_size, activations[i] if i < len(activations) else "relu")
            for i in range(num_mixer_layers)
        ])

        self.fourier_head = CompleteFourierHead(latent_dim, num_modes, init_scale, spacing_mode, alpha)
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
            "a0": self.fourier_head.a0.item(),
            "spacing_mode": self.fourier_head.spacing_mode,
            "alpha": self.fourier_head.alpha,
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


def run_single(args, spacing_mode, alpha, activations, label):
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

    model = SpectralV7(
        img_size=img_size, in_ch=in_ch, latent_dim=args.latent_dim,
        num_modes=args.num_modes, num_mixer_layers=args.num_mixer_layers,
        num_classes=num_classes, head_hidden=args.head_hidden,
        init_scale=args.init_scale, spacing_mode=spacing_mode, alpha=alpha,
        activations=activations,
    ).to(device)

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
    return best_acc, count_params(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "fashion", "cifar10"])
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--num_modes", type=int, default=64)
    parser.add_argument("--num_mixer_layers", type=int, default=2)
    parser.add_argument("--init_scale", type=float, default=2.0)
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    configs = [
        ("uniform", 1.0, ["relu", "relu"], "v6-baseline"),
        ("power", 0.5, ["relu", "relu"], "power-0.5"),
        ("power", 0.75, ["relu", "relu"], "power-0.75"),
        ("power", 1.5, ["relu", "relu"], "power-1.5"),
        ("power", 2.0, ["relu", "relu"], "power-2.0"),
        ("uniform", 1.0, ["square", "relu"], "mixed-activ"),
        ("power", 0.75, ["square", "relu"], "power-0.75+mix"),
    ]

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'} | Dataset: {args.dataset}")
    print(f"Patchless global FFT | Complete Fourier basis | Non-uniform spacing\n")
    print("="*70)
    print(f"{'Config':<20} {'Spacing':<12} {'Alpha':<8} {'Best Acc':<12} {'Params':<10}")
    print("-"*70)

    for spacing, alpha, acts, label in configs:
        try:
            acc, params = run_single(args, spacing, alpha, acts, label)
            print(f"{label:<20} {spacing:<12} {alpha:<8.2f} {acc:<12.4f} {params:<10}")
        except Exception as e:
            print(f"{label:<20} FAILED: {e}")

    print("="*70)


if __name__ == "__main__":
    main()
