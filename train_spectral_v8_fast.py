"""
Spectral v8 Fast: Generic N-D with optimized paths for rank 1/2/3
=================================================================
- Channel mixing: Conv1d/2d/3d for common ranks (BLAS-optimized).
  Falls back to einsum only for rank >= 4.
- FFT: rfft/rfft2/rfftn matched to actual rank.
- Adaptive learnable frequency grid.
- Square wave activation, 4 layers, constant LR 1e-2.
"""
import torch
import torch.nn as nn
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
    "train_images": "train-images-idx3-ubyte.gz", "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",   "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_mnist(data_dir):
    files = {
        "train_images": "train-images-idx3-ubyte.gz", "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",   "test_labels": "t10k-labels-idx1-ubyte.gz",
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
    x_train, y_train = [], []
    for i in range(1, 6):
        with open(os.path.join(extracted, f"data_batch_{i}"), "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        x_train.append(batch[b"data"])
        y_train.extend(batch[b"labels"])
    x_train = np.concatenate(x_train, axis=0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_train = np.array(y_train, dtype=np.int64)
    with open(os.path.join(extracted, "test_batch"), "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    x_test = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_test = np.array(batch[b"labels"], dtype=np.int64)
    return torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(x_test), torch.from_numpy(y_test)


# ---------------------------------------------------------------------------
# Fast channel projection: ConvNd for rank <= 3, einsum for rank >= 4
# ---------------------------------------------------------------------------
class FastChannelProjection(nn.Module):
    """
    Uses native PyTorch ConvNd for speed on 1D/2D/3D.
    Falls back to einsum for rank >= 4.
    """
    def __init__(self, in_channels, out_channels, spatial_rank):
        super().__init__()
        self.rank = spatial_rank
        if spatial_rank == 1:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        elif spatial_rank == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        elif spatial_rank == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        else:
            self.conv = None
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels) * 0.02)
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        if self.conv is not None:
            return self.conv(x)
        # generic einsum fallback for rank >= 4
        rank = x.ndim - 2
        letters = "".join(chr(ord("d") + i) for i in range(rank))
        ein_in = "bc" + letters
        ein_w = "Dc"
        ein_out = "bD" + letters
        out = torch.einsum(f"{ein_in},{ein_w}->{ein_out}", x, self.weight)
        shape = [1, -1] + [1] * rank
        return out + self.bias.view(*shape)


# ---------------------------------------------------------------------------
# Activations
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


class ChebyshevActivation(nn.Module):
    def __init__(self, order=4):
        super().__init__()
        self.order = order
        self.weights = nn.Parameter(torch.randn(order + 1) * 0.1)

    def forward(self, x):
        x = torch.tanh(x)
        T = [torch.ones_like(x), x]
        for n in range(2, self.order + 1):
            T.append(2 * x * T[-1] - T[-2])
        return sum(self.weights[n] * T[n] for n in range(self.order + 1))


class FMActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_w1 = nn.Parameter(torch.tensor(0.0))
        self.log_w2 = nn.Parameter(torch.tensor(0.0))
        self.log_A = nn.Parameter(torch.tensor(0.0))
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        w1 = torch.exp(self.log_w1); w2 = torch.exp(self.log_w2)
        A = torch.exp(self.log_A); alpha = torch.exp(self.log_alpha)
        return torch.tanh(alpha * torch.sin(w1 * x + A * torch.sin(w2 * x)))


# ---------------------------------------------------------------------------
# Generic N-D Spectral Mixer with FFT fast paths
# ---------------------------------------------------------------------------
class GenericSpectralMixer(nn.Module):
    def __init__(self, channels, spatial_shape, activation="square"):
        super().__init__()
        self.channels = channels
        self.spatial_shape = tuple(spatial_shape)
        self.rank = len(spatial_shape)
        self.spatial_axes = list(range(2, 2 + self.rank))

        # Infer FFT shape
        dummy = torch.zeros(1, 1, *spatial_shape)
        if self.rank == 1:
            dummy_fft = torch.fft.rfft(dummy, dim=2)
        elif self.rank == 2:
            dummy_fft = torch.fft.rfft2(dummy, dim=(2, 3))
        else:
            dummy_fft = torch.fft.rfftn(dummy, dim=self.spatial_axes)
        self.fft_shape = dummy_fft.shape[2:]

        self.gain_real = nn.Parameter(torch.ones(channels, *self.fft_shape) * 0.5)
        self.gain_imag = nn.Parameter(torch.zeros(channels, *self.fft_shape))
        self.norm = nn.GroupNorm(num_groups=max(1, channels // 4), num_channels=channels)

        act_map = {
            "none": nn.Identity(), "relu": nn.ReLU(), "gelu": nn.GELU(),
            "swish": nn.SiLU(), "elu": nn.ELU(), "softplus": nn.Softplus(),
            "square": LearnableSquareWave(steepness=10.0),
            "chebyshev": ChebyshevActivation(order=4),
            "fm": FMActivation(),
        }
        self.activation = act_map.get(activation, nn.Identity())

    def _fft(self, x):
        if self.rank == 1:
            return torch.fft.rfft(x, dim=2)
        elif self.rank == 2:
            return torch.fft.rfft2(x, dim=(2, 3))
        return torch.fft.rfftn(x, dim=self.spatial_axes)

    def _ifft(self, x):
        if self.rank == 1:
            return torch.fft.irfft(x, n=self.spatial_shape[0], dim=2)
        elif self.rank == 2:
            return torch.fft.irfft2(x, s=self.spatial_shape, dim=(2, 3))
        return torch.fft.irfftn(x, s=self.spatial_shape, dim=self.spatial_axes)

    def forward(self, x):
        x_fft = self._fft(x)
        gain = torch.view_as_complex(torch.stack([self.gain_real, self.gain_imag], dim=-1))
        x_filtered = x_fft * gain.unsqueeze(0)
        x_out = self._ifft(x_filtered)
        x_out = self.activation(x_out)
        return self.norm(x_out + x)


# ---------------------------------------------------------------------------
# Adaptive Fourier Head
# ---------------------------------------------------------------------------
def _init_split_grid(num_modes):
    n_dense = int((num_modes + 1) * 0.6)
    n_rest = (num_modes + 1) - n_dense
    dense = torch.linspace(0.0, 1.0, steps=n_dense + 1)[:-1]
    rest = torch.linspace(1.0, float(num_modes), steps=n_rest)
    return torch.cat([dense, rest])


class AdaptiveFourierHead(nn.Module):
    def __init__(self, latent_dim, num_modes=64, init_scale=2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.proj_weight = nn.Parameter(torch.randn(latent_dim) * 0.1)
        self.proj_bias = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.a0 = nn.Parameter(torch.zeros(1))
        self.a_n = nn.Parameter(torch.randn(num_modes) * 0.1)
        self.b_n = nn.Parameter(torch.randn(num_modes) * 0.1)
        init_grid = _init_split_grid(num_modes)
        self.log_harmonic_n = nn.Parameter(torch.log(init_grid[1:] + 1e-3))

    @property
    def harmonic_n(self):
        return torch.exp(self.log_harmonic_n)

    def forward(self, z):
        z_1d = torch.matmul(z, self.proj_weight) + self.proj_bias
        freqs = 2.0 * np.pi * self.harmonic_n / (self.scale.abs() + 1e-6)
        proj = z_1d.unsqueeze(1) * freqs.unsqueeze(0)
        cos_terms = self.a_n * torch.cos(proj)
        sin_terms = self.b_n * torch.sin(proj)
        fourier_out = self.a0 + cos_terms.sum(dim=1) + sin_terms.sum(dim=1)
        fourier_out = fourier_out / (self.num_modes + 1)
        return fourier_out.unsqueeze(-1), z


# ---------------------------------------------------------------------------
# Spectral v8 Fast Model
# ---------------------------------------------------------------------------
class SpectralV8Fast(nn.Module):
    def __init__(self, spatial_shape, in_channels, latent_dim=16, num_modes=64,
                 num_mixer_layers=4, num_classes=10, head_hidden=128,
                 init_scale=2.0, activation="square"):
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_shape = tuple(spatial_shape)
        self.channel_proj = FastChannelProjection(in_channels, latent_dim, len(spatial_shape))

        self.mixers = nn.ModuleList([
            GenericSpectralMixer(latent_dim, spatial_shape, activation)
            for _ in range(num_mixer_layers)
        ])

        self.fourier_head = AdaptiveFourierHead(latent_dim, num_modes, init_scale)
        self.classifier = nn.Sequential(
            nn.Linear(1 + latent_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x):
        z = self.channel_proj(x)
        for mixer in self.mixers:
            z = mixer(z)
        z = z.mean(dim=list(range(2, z.ndim)))
        fourier_scalar, z_full = self.fourier_head(z)
        features = torch.cat([fourier_scalar, z_full], dim=-1)
        logits = self.classifier(features)
        return logits

    def get_fourier_info(self):
        with torch.no_grad():
            grid = self.fourier_head.harmonic_n
            return {
                "scale": self.fourier_head.scale.item(),
                "proj_weight_norm": self.fourier_head.proj_weight.norm().item(),
                "a0": self.fourier_head.a0.item(),
                "min_h": grid.min().item(),
                "max_h": grid.max().item(),
                "mean_h": grid.mean().item(),
            }


# ---------------------------------------------------------------------------
# Training utilities
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "fashion", "cifar10"])
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--num_modes", type=int, default=64)
    parser.add_argument("--num_mixer_layers", type=int, default=4)
    parser.add_argument("--activation", type=str, default="square",
                        choices=["none", "relu", "square", "gelu", "chebyshev", "fm", "swish", "elu", "softplus"])
    parser.add_argument("--init_scale", type=float, default=2.0)
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
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
        spatial_shape = (28, 28); in_ch = 1; num_classes = 10
    elif args.dataset == "fashion":
        _download_fashion(os.path.join(args.data_dir, "fashion"))
        x_tr, y_tr = _load_fashion(os.path.join(args.data_dir, "fashion"), "train")
        x_te, y_te = _load_fashion(os.path.join(args.data_dir, "fashion"), "test")
        spatial_shape = (28, 28); in_ch = 1; num_classes = 10
    elif args.dataset == "cifar10":
        x_tr, y_tr, x_te, y_te = _load_cifar10(os.path.join(args.data_dir, "cifar10"))
        spatial_shape = (32, 32); in_ch = 3; num_classes = 10

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=args.batch_size, shuffle=False)

    model = SpectralV8Fast(
        spatial_shape=spatial_shape, in_channels=in_ch, latent_dim=args.latent_dim,
        num_modes=args.num_modes, num_mixer_layers=args.num_mixer_layers,
        num_classes=num_classes, head_hidden=args.head_hidden,
        init_scale=args.init_scale, activation=args.activation,
    ).to(device)

    print(f"Device: {device} | Dataset: {args.dataset}")
    print(f"Spatial dims: {spatial_shape} | Layers: {args.num_mixer_layers}")
    print(f"Constant LR: {args.lr} | Adaptive freq grid | Activation: {args.activation}\n")
    print("="*60)
    print("SPECTRAL v8 FAST — Optimized paths for rank <= 3")
    print("="*60)
    print(f"Latent dim: {args.latent_dim} | Mixer layers: {args.num_mixer_layers}")
    print(f"Activation: {args.activation} | Total parameters: {count_params(model)}")
    print("-"*60)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = evaluate(model, test_loader, criterion, device)
        if va > best_acc:
            best_acc = va
        if epoch <= 3 or epoch % 5 == 0 or epoch == args.epochs:
            info = model.get_fourier_info()
            print(f"Epoch {epoch:02d} | Train Acc: {ta:.4f} | Test Acc: {va:.4f} | "
                  f"Scale: {info['scale']:.3f} | "
                  f"Harmonics: [{info['min_h']:.3f}, {info['mean_h']:.3f}, {info['max_h']:.3f}]")
    print(f"Best accuracy: {best_acc:.4f}\n")

    if args.run_baseline:
        baseline = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(x_tr.shape[1:]), 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        ).to(device)
        print("="*60); print("BASELINE MLP"); print("="*60)
        print(f"Total parameters: {count_params(baseline)}"); print("-"*60)
        opt_b = torch.optim.Adam(baseline.parameters(), lr=args.lr)
        best_b = 0.0
        for epoch in range(1, args.epochs + 1):
            tl, ta = train_epoch(baseline, train_loader, opt_b, criterion, device)
            vl, va = evaluate(baseline, test_loader, criterion, device)
            if va > best_b: best_b = va
            if epoch <= 3 or epoch % 5 == 0 or epoch == args.epochs:
                print(f"Epoch {epoch:02d} | Train Acc: {ta:.4f} | Test Acc: {va:.4f}")
        print(f"Best Baseline accuracy: {best_b:.4f}\n")


if __name__ == "__main__":
    main()
