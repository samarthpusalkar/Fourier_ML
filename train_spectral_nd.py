"""
Spectral-Locality Network: True N-Dimensional FFT Processing
==============================================================

Corrects the fundamental spatial structure mistake in prior versions.

Problem with V1-V6: Patches were flattened to a 1D sequence, and spectral mixing
was 1D FFT across the flattened patch sequence. This destroys 2D spatial structure.

Solution: Perform N-D FFT over the actual spatial dimensions of the data.

For images (2D spatial data):
  Input: (B, C, H, W)
  After patch embedding: (B, latent_dim, H/ph, W/pw)  [keep spatial grid!]
  Spectral Mixer: 2D FFT over (H/ph, W/pw) spatial dimensions
  Learnable gains: per frequency bin in the 2D frequency plane
  This naturally captures horizontal/vertical/diagonal orientations as frequency components.

For tabular (0D):
  No spatial dimensions, skip mixer, go straight to Fourier head.

For sequences (1D):
  1D FFT over sequence length.

This is the signal processing approach the user correctly identified.
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


def _load_iris():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    X, y = load_iris(return_X_y=True)
    X = X.astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return (torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64)),
            torch.from_numpy(x_test), torch.from_numpy(y_test.astype(np.int64)))


# ---------------------------------------------------------------------------
# N-Dimensional Spatial Patch Embed
# ---------------------------------------------------------------------------
class NDSpatialPatchEmbed(nn.Module):
    """
    Converts input to a spatial grid of patches with learned encoding.
    Output preserves N-D spatial structure: (B, latent_dim, *spatial_dims).
    """
    def __init__(self, patch_size=(4, 4), in_ch=1, latent_dim=16):
        super().__init__()
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.latent_dim = latent_dim
        patch_dim = patch_size[0] * patch_size[1] * in_ch
        self.encoder = nn.Linear(patch_dim, latent_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        # unfold into patches: (B, C, H//ph, W//pw, ph, pw)
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)
        x = x.contiguous().view(B, C, H // ph, W // pw, ph, pw)
        # permute to (B, H', W', C, ph, pw)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        # flatten patch pixels: (B, H', W', C*ph*pw)
        x = x.view(B, H // ph, W // pw, -1)
        # encode each patch to latent_dim: (B, H', W', latent_dim)
        B_prime, H_prime, W_prime, _ = x.shape
        x = x.view(B_prime * H_prime * W_prime, -1)
        x = self.encoder(x)
        x = x.view(B_prime, H_prime, W_prime, self.latent_dim)
        # transpose to (B, latent_dim, H', W') for FFT
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# ---------------------------------------------------------------------------
# N-Dimensional Spectral Mixer
# ---------------------------------------------------------------------------
class NDSpectralMixer(nn.Module):
    """
    Performs N-D FFT over spatial dimensions, applies learnable complex gains,
    then inverse N-D FFT.  This is the generic spatial feature filtering layer.

    For images: input is (B, latent_dim, H, W) --> 2D FFT over (H, W).
    """
    def __init__(self, latent_dim, spatial_shape, ndim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.ndim = ndim

        # For 2D case: fftfreq returns normalized frequencies, gains per (freq_h, freq_w)
        if ndim == 2:
            h, w = spatial_shape
            self.freq_shape = (h, w // 2 + 1)  # rfft2 output shape
            self.gain_real = nn.Parameter(torch.ones(latent_dim, *self.freq_shape) * 0.5)
            self.gain_imag = nn.Parameter(torch.zeros(latent_dim, *self.freq_shape))
        elif ndim == 1:
            n = spatial_shape[0]
            self.freq_shape = (n // 2 + 1,)
            self.gain_real = nn.Parameter(torch.ones(latent_dim, *self.freq_shape) * 0.5)
            self.gain_imag = nn.Parameter(torch.zeros(latent_dim, *self.freq_shape))

        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z):
        # z: (B, latent_dim, *spatial_dims)
        if self.ndim == 2:
            # 2D FFT over spatial dims: (B, latent_dim, H, W)
            z_fft = torch.fft.rfft2(z, dim=(-2, -1))  # (B, latent_dim, H, W//2+1)
            # Apply learnable complex gains
            gain = torch.view_as_complex(torch.stack([self.gain_real, self.gain_imag], dim=-1))
            # gain shape: (latent_dim, H, W//2+1)
            z_filtered = z_fft * gain.unsqueeze(0)  # broadcast over batch
            # Inverse 2D FFT - need original spatial shape
            original_shape = z.shape[-2:]
            z_out = torch.fft.irfft2(z_filtered, s=original_shape, dim=(-2, -1))
        elif self.ndim == 1:
            z_fft = torch.fft.rfft(z, dim=-1)
            gain = torch.view_as_complex(torch.stack([self.gain_real, self.gain_imag], dim=-1))
            z_filtered = z_fft * gain.unsqueeze(0)
            original_len = z.shape[-1]
            z_out = torch.fft.irfft(z_filtered, n=original_len, dim=-1)

        # z_out: (B, latent_dim, *spatial_dims)
        # Residual + LayerNorm (operate on channel dim)
        z_perm = z_out.permute(0, *list(range(2, 2 + self.ndim)), 1).contiguous()
        z_orig_perm = z.permute(0, *list(range(2, 2 + self.ndim)), 1).contiguous()
        # Flatten spatial for LayerNorm
        old_shape = z_perm.shape
        z_flat = z_perm.view(-1, self.latent_dim)
        z_orig_flat = z_orig_perm.view(-1, self.latent_dim)
        z_normed = self.norm(z_flat + z_orig_flat)
        z_perm_out = z_normed.view(old_shape)
        # permute back
        perm = [0, 1 + self.ndim] + list(range(1, 1 + self.ndim))
        z_final = z_perm_out.permute(*perm).contiguous()
        return z_final


# ---------------------------------------------------------------------------
# Fixed-integer-harmonic Fourier with learnable scale
# ---------------------------------------------------------------------------
class GenericFourierHead(nn.Module):
    def __init__(self, latent_dim, num_modes, init_scale=2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.proj_weight = nn.Parameter(torch.randn(latent_dim) * 0.1)
        self.proj_bias = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.register_buffer('harmonic_n', torch.arange(num_modes + 1, dtype=torch.float32))

    def forward(self, z):
        # z: (B, latent_dim) -- pooled spatial representation
        z_1d = torch.matmul(z, self.proj_weight) + self.proj_bias
        freqs = 2.0 * np.pi * self.harmonic_n / (self.scale.abs() + 1e-6)
        proj = z_1d.unsqueeze(1) * freqs.unsqueeze(0)
        cos_terms = torch.cos(proj)
        sin_terms = torch.sin(proj)
        fourier_scalar = cos_terms.sum(dim=-1, keepdim=True) + sin_terms.sum(dim=-1, keepdim=True)
        fourier_scalar = fourier_scalar / (self.num_modes + 1)
        return fourier_scalar, z


# ---------------------------------------------------------------------------
# Latent Energy Gate
# ---------------------------------------------------------------------------
class LatentEnergyGate(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, z):
        # z: (B, latent_dim, *spatial)
        gate = torch.sigmoid(self.gate_logits)
        # unsqueeze gate for spatial dims
        for _ in range(z.dim() - 2):
            gate = gate.unsqueeze(-1)
        return z * gate


# ---------------------------------------------------------------------------
# Unified N-D Spectral Model
# ---------------------------------------------------------------------------
class NDSpectralModel(nn.Module):
    """
    True N-D spectral model.  Spatial FFT respects actual image dimensions.
    """
    def __init__(self, input_type, img_size=None, patch_size=None, in_ch=None,
                 input_dim=None, latent_dim=16, num_modes=64, num_mixer_layers=2,
                 num_classes=10, head_hidden=128, init_scale=2.0, ndim=2):
        super().__init__()
        self.input_type = input_type
        self.ndim = ndim
        self.latent_dim = latent_dim

        if input_type == "image":
            self.embed = NDSpatialPatchEmbed(patch_size=patch_size, in_ch=in_ch, latent_dim=latent_dim)
            ph, pw = patch_size
            spatial_shape = (img_size[0] // ph, img_size[1] // pw)
            self.mixers = nn.ModuleList([
                NDSpectralMixer(latent_dim, spatial_shape, ndim=2)
                for _ in range(num_mixer_layers)
            ])
        else:
            self.encoder = nn.Linear(input_dim, latent_dim)
            self.embed = None
            self.mixers = nn.ModuleList([])

        self.gate = LatentEnergyGate(latent_dim)
        self.fourier_head = GenericFourierHead(latent_dim, num_modes, init_scale)
        self.classifier = nn.Sequential(
            nn.Linear(1 + latent_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x):
        if self.input_type == "image":
            # (B, latent_dim, H', W')
            z = self.embed(x)
            z = self.gate(z)
            for mixer in self.mixers:
                z = mixer(z)
                z = self.gate(z)
            # Global average pool over spatial dimensions
            z = z.mean(dim=[-2, -1])  # (B, latent_dim)
        else:
            z = self.encoder(x)
            z = self.gate(z.unsqueeze(-1)).squeeze(-1)  # hack for 1D tabular

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
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10", "iris"])
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--num_modes", type=int, default=64)
    parser.add_argument("--num_mixer_layers", type=int, default=2)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[4, 4])
    parser.add_argument("--init_scale", type=float, default=2.0)
    parser.add_argument("--head_hidden", type=int, default=128)
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
    print(f"Architecture: TRUE N-D spatial FFT (2D FFT over patch grid)\n")

    if args.dataset == "mnist":
        _download_mnist(os.path.join(args.data_dir, "mnist"))
        x_tr, y_tr = _load_mnist(os.path.join(args.data_dir, "mnist"), "train")
        x_te, y_te = _load_mnist(os.path.join(args.data_dir, "mnist"), "test")
        input_type = "image"
        img_size = (28, 28); in_ch = 1; num_classes = 10; ndim = 2
    elif args.dataset == "cifar10":
        x_tr, y_tr, x_te, y_te = _load_cifar10(os.path.join(args.data_dir, "cifar10"))
        input_type = "image"
        img_size = (32, 32); in_ch = 3; num_classes = 10; ndim = 2
    elif args.dataset == "iris":
        x_tr, y_tr, x_te, y_te = _load_iris()
        input_type = "flat"
        img_size = None; in_ch = None; num_classes = 3; ndim = 0

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=args.batch_size, shuffle=False
    )

    model = NDSpectralModel(
        input_type=input_type, img_size=img_size, patch_size=tuple(args.patch_size),
        in_ch=in_ch, input_dim=x_tr.shape[1] if input_type=="flat" else None,
        latent_dim=args.latent_dim, num_modes=args.num_modes,
        num_mixer_layers=args.num_mixer_layers, num_classes=num_classes,
        head_hidden=args.head_hidden, init_scale=args.init_scale, ndim=ndim,
    ).to(device)

    print("="*60)
    print("N-D SPECTRAL MODEL")
    print("="*60)
    print(f"Latent dim: {args.latent_dim}")
    print(f"FFT type: {'2D FFT over patch grid' if input_type == 'image' else 'No FFT'}")
    print(f"Total parameters: {count_params(model)}")
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
    print(f"Best N-D Spectral accuracy: {best_acc:.4f}\n")

    if args.run_baseline:
        baseline = StandardMLP(input_dim=np.prod(x_tr.shape[1:]) if input_type=="image" else x_tr.shape[1],
                               hidden=[256,128] if input_type=="image" else [64,32], num_classes=num_classes).to(device)
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
