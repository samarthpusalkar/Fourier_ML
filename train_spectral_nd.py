"""
Spectral-Locality Network for N-D Tensors
=========================================
Extends the SL-Net ideas to tensor inputs (images, patches, tokens)
without flattening everything to 1D.

Key architectural decisions:
  1. PatchEmbed / tokenization preserves spatial structure.
  2. Each patch is projected to a SHARED low-D latent space (learnable encoder per patch).
  3. Spectral Token Mixer (STM): replaces Q/K/V attention with a learned
     frequency-domain filter over the sequence of patch latents.
  4. Fourier Classification Head: same learnable sinusoidal layer as before,
     but operates on the globally-pooled spectral representation.

This is NOT an LLM, but it answers whether spectral methods can replace
attention for structured data.  The answer: yes, for classification,
via learned frequency-domain token mixing (similar to FNet but with
learnable per-frequency gains rather than a plain FFT pass).
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


# ---------------------------------------------------------------------------
# Minimal MNIST loader (same as before)
# ---------------------------------------------------------------------------
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def _download_mnist(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for key, fname in MNIST_FILES.items():
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
    images = images.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    with gzip.open(lbl_path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    labels = labels.astype(np.int64)
    return torch.from_numpy(images), torch.from_numpy(labels)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, reshape=None):
        self.reshape = reshape
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.reshape is not None:
            x = x.view(self.reshape)
        return x, self.labels[idx]


# ---------------------------------------------------------------------------
# 1 Patch embedding: treats image as a sequence of local patches
#    MNIST 28x28 -> 7x7 grid of 4x4 patches (49 tokens, 16 dims each)
# ---------------------------------------------------------------------------
class PatchEmbed1D(nn.Module):
    """
    Converts flat image -> (B, num_patches, patch_dim) by reshaping.
    No convolutions; purely a view/reshape, so spatial meaning is preserved.
    """
    def __init__(self, img_size=(28, 28), patch_size=(4, 4), in_ch=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_dim = patch_size[0] * patch_size[1] * in_ch

    def forward(self, x):
        # x: (B, C*H*W) or (B, C, H, W)
        if x.dim() == 2:
            B = x.size(0)
            x = x.view(B, 1, self.img_size[0], self.img_size[1])
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        # unfold to patches: (B, C, H/ph, ph, W/pw, pw) -> (B, num_patches, patch_dim)
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)
        x = x.contiguous().view(B, C, H // ph, W // pw, ph, pw)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * ph * pw)
        return x


# ---------------------------------------------------------------------------
# 2 Per-patch latent encoder (shared weights across all patches)
# ---------------------------------------------------------------------------
class PatchLatentEncoder(nn.Module):
    """
    Shared MLP that compresses each patch independently to a low-D latent.
    Replaces the global 1D collapse with local, translation-equivariant compression.
    """
    def __init__(self, patch_dim, latent_dim=8, hidden_dims=[32]):
        super().__init__()
        layers = []
        prev = patch_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, num_patches, patch_dim)
        B, N, D = x.shape
        x = x.view(B * N, D)
        z = self.net(x)
        z = z.view(B, N, -1)
        return z


# ---------------------------------------------------------------------------
# 3 Spectral Token Mixer (STM)
#    Replaces Q/K/V self-attention.
#    Applies FFT across the patch sequence, multiplies by learnable complex mask,
#    then IFFT back.  This mixes information between patches in the frequency
#    domain, analogous to FNet but with learned per-frequency gains.
# ---------------------------------------------------------------------------
class SpectralTokenMixer(nn.Module):
    """
    Learnable frequency-domain filter over the token (patch) sequence.
    Forward:
      1. FFT over sequence dimension (N = num_patches)
      2. Multiply each frequency bin by a learnable complex gain (A + iB)
      3. IFFT back to token space
      4. Residual connection + LayerNorm
    """
    def __init__(self, latent_dim, num_patches):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_patches = num_patches

        # Learnable real/imaginary gains for each FFT bin and each latent channel
        # Shape: (latent_dim, num_patches // 2 + 1) for real and imag parts
        n_rfft = num_patches // 2 + 1
        self.gain_real = nn.Parameter(torch.ones(latent_dim, n_rfft) * 0.5)
        self.gain_imag = nn.Parameter(torch.zeros(latent_dim, n_rfft))
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z):
        # z: (B, N, latent_dim)
        # Transpose to (B, latent_dim, N) for per-channel FFT
        z_t = z.transpose(1, 2)  # (B, latent_dim, N)

        # RFFT -> complex representation
        z_fft = torch.fft.rfft(z_t, dim=-1)  # (B, latent_dim, n_rfft)

        # Apply learnable complex gain
        gain = torch.view_as_complex(
            torch.stack([self.gain_real, self.gain_imag], dim=-1)
        )  # (latent_dim, n_rfft)
        z_filtered = z_fft * gain.unsqueeze(0)  # broadcast over batch

        # IRFFT back
        z_out = torch.fft.irfft(z_filtered, n=self.num_patches, dim=-1)  # (B, latent_dim, N)
        z_out = z_out.transpose(1, 2)  # (B, N, latent_dim)

        # Residual + norm
        return self.norm(z + z_out)


# ---------------------------------------------------------------------------
# 4 Fourier Classification Head (on globally pooled latent)
# ---------------------------------------------------------------------------
class FourierHead(nn.Module):
    """
    Global average pool over tokens, then learnable sinusoidal classification.
    Same spirit as the 1D Fourier layer, but now the input is a pooled vector.
    """
    def __init__(self, latent_dim, num_modes=64, num_classes=10):
        super().__init__()
        self.num_modes = num_modes
        self.latent_dim = latent_dim

        self.frequencies = nn.Parameter(torch.randn(num_modes, latent_dim) * 0.1)
        self.A = nn.Parameter(torch.randn(num_modes) * 0.01)
        self.B = nn.Parameter(torch.randn(num_modes) * 0.01)
        self.dc = nn.Parameter(torch.zeros(1))

        # Classifier on the single scalar output of the Fourier layer
        # plus optionally the raw pooled latent (for residual capacity)
        self.classifier = nn.Sequential(
            nn.Linear(1 + latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, z_pooled):
        # z_pooled: (B, latent_dim)
        proj = torch.matmul(z_pooled, self.frequencies.T)  # (B, num_modes)
        cos_feat = torch.cos(proj)
        sin_feat = torch.sin(proj)
        fourier_scalar = self.dc + torch.sum(
            self.A * cos_feat + self.B * sin_feat, dim=-1, keepdim=True
        )  # (B, 1)

        features = torch.cat([fourier_scalar, z_pooled], dim=-1)
        logits = self.classifier(features)
        return logits

    def get_spectral_params(self):
        return {
            "frequencies": self.frequencies.detach().cpu().numpy(),
            "A": self.A.detach().cpu().numpy(),
            "B": self.B.detach().cpu().numpy(),
            "dc": self.dc.detach().cpu().item(),
        }


# ---------------------------------------------------------------------------
# Full model assembly
# ---------------------------------------------------------------------------
class SpectralNDClassifier(nn.Module):
    """
    N-D tensor -> PatchEmbed -> Per-patch latent encoder -> Spectral Token Mixer -> Pool -> Fourier classification head.
    """
    def __init__(
        self,
        img_size=(28, 28),
        patch_size=(4, 4),
        in_ch=1,
        latent_dim=8,
        patch_hidden=[32],
        num_mixer_layers=2,
        num_modes=64,
        num_classes=10,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed1D(img_size, patch_size, in_ch)
        num_patches = self.patch_embed.num_patches
        patch_dim = self.patch_embed.patch_dim

        self.encoder = PatchLatentEncoder(patch_dim, latent_dim, patch_hidden)

        self.mixers = nn.ModuleList([
            SpectralTokenMixer(latent_dim, num_patches)
            for _ in range(num_mixer_layers)
        ])

        self.head = FourierHead(latent_dim, num_modes, num_classes)

    def forward(self, x):
        # x: (B, C, H, W) or (B, C*H*W)
        patches = self.patch_embed(x)       # (B, num_patches, patch_dim)
        z = self.encoder(patches)           # (B, num_patches, latent_dim)
        for mixer in self.mixers:
            z = mixer(z)
        z_pooled = z.mean(dim=1)            # (B, latent_dim)
        logits = self.head(z_pooled)
        return logits

    def get_fourier_params(self):
        return self.head.get_spectral_params()

    def get_token_mixer_gains(self):
        """Returns the learned real/imaginary gains for each FFT bin."""
        gains = []
        for i, mixer in enumerate(self.mixers):
            gains.append({
                "layer": i,
                "gain_real": mixer.gain_real.detach().cpu().numpy().tolist(),
                "gain_imag": mixer.gain_imag.detach().cpu().numpy().tolist(),
            })
        return gains


# ---------------------------------------------------------------------------
# Training helpers (same interface)
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
# Baseline standard MLP for comparison
# ---------------------------------------------------------------------------
class StandardMLP(nn.Module):
    def __init__(self, input_dim=784, hidden=[256, 128], num_classes=10):
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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Spectral N-D Classifier on MNIST")
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[4, 4])
    parser.add_argument("--num_mixer_layers", type=int, default=2)
    parser.add_argument("--num_modes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data/mnist")
    parser.add_argument("--run_baseline", action="store_true",
                        help="Also train a standard MLP for parameter comparison")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    _download_mnist(args.data_dir)
    x_train, y_train = _load_mnist(args.data_dir, "train")
    x_test, y_test = _load_mnist(args.data_dir, "test")

    # For N-D model, keep images as (C, H, W)
    train_ds_nd = SimpleDataset(x_train, y_train, reshape=(1, 28, 28))
    test_ds_nd = SimpleDataset(x_test, y_test, reshape=(1, 28, 28))
    train_loader_nd = torch.utils.data.DataLoader(train_ds_nd, batch_size=args.batch_size, shuffle=True)
    test_loader_nd = torch.utils.data.DataLoader(test_ds_nd, batch_size=args.batch_size, shuffle=False)

    # --- Spectral N-D model ---
    model_nd = SpectralNDClassifier(
        img_size=(28, 28),
        patch_size=tuple(args.patch_size),
        in_ch=1,
        latent_dim=args.latent_dim,
        patch_hidden=[32],
        num_mixer_layers=args.num_mixer_layers,
        num_modes=args.num_modes,
        num_classes=10,
    ).to(device)

    print("=" * 60)
    print("Spectral N-D Classifier")
    print("=" * 60)
    print(model_nd)
    print(f"Total parameters: {count_params(model_nd)}")
    print("-" * 60)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_nd.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model_nd, train_loader_nd, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model_nd, test_loader_nd, criterion, device)
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            ckpt = {
                "state_dict": model_nd.state_dict(),
                "fourier_params": model_nd.get_fourier_params(),
                "mixer_gains": model_nd.get_token_mixer_gains(),
            }
            torch.save(ckpt, "best_spectral_nd.pt")
        print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Best Spectral N-D accuracy: {best_acc:.4f}\n")

    # --- Optional baseline comparison ---
    if args.run_baseline:
        train_ds_flat = SimpleDataset(x_train, y_train)
        test_ds_flat = SimpleDataset(x_test, y_test)
        train_loader_flat = torch.utils.data.DataLoader(train_ds_flat, batch_size=args.batch_size, shuffle=True)
        test_loader_flat = torch.utils.data.DataLoader(test_ds_flat, batch_size=args.batch_size, shuffle=False)

        baseline = StandardMLP(input_dim=784, hidden=[256, 128], num_classes=10).to(device)
        print("=" * 60)
        print("Standard MLP Baseline")
        print("=" * 60)
        print(baseline)
        print(f"Total parameters: {count_params(baseline)}")
        print("-" * 60)

        optimizer_b = torch.optim.Adam(baseline.parameters(), lr=args.lr)
        scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=args.epochs)
        best_b = 0.0
        for epoch in range(1, args.epochs + 1):
            tl, ta = train_epoch(baseline, train_loader_flat, optimizer_b, criterion, device)
            vl, va = evaluate(baseline, test_loader_flat, criterion, device)
            scheduler_b.step()
            if va > best_b:
                best_b = va
            print(f"Epoch {epoch:02d} | Train Acc: {ta:.4f} | Test Acc: {va:.4f}")
        print(f"Best Baseline accuracy: {best_b:.4f}")

    # --- Inspect learned Fourier parameters ---
    print("\n" + "=" * 60)
    print("Learned Fourier Head Parameters (by energy)")
    print("=" * 60)
    fp = model_nd.get_fourier_params()
    freqs = fp["frequencies"]   # (num_modes, latent_dim)
    A = fp["A"]
    B = fp["B"]
    energy = np.sqrt(A**2 + B**2)
    idx = np.argsort(-energy)
    for rank, i in enumerate(idx[:12], 1):
        f_norm = np.linalg.norm(freqs[i])
        print(
            f"  Rank {rank:2d} | Mode {i:2d} | freq_norm={f_norm:.3f} | "
            f"energy={energy[i]:.3f} | A={A[i]:7.3f} | B={B[i]:7.3f}"
        )
    print(f"  DC offset = {fp['dc']:.4f}")

    # Save human-readable spectral report
    report = {
        "model": "SpectralNDClassifier",
        "config": vars(args),
        "total_params": count_params(model_nd),
        "best_accuracy": float(best_acc),
        "fourier_head": {
            "dc": float(fp["dc"]),
            "top_modes": [],
        },
        "token_mixer_gains": model_nd.get_token_mixer_gains(),
    }
    for rank, i in enumerate(idx[:16], 1):
        report["fourier_head"]["top_modes"].append({
            "rank": rank,
            "mode_index": int(i),
            "freq_vector": freqs[i].tolist(),
            "freq_norm": float(np.linalg.norm(freqs[i])),
            "A": float(A[i]),
            "B": float(B[i]),
            "energy": float(energy[i]),
            "phase_rad": float(np.arctan2(-B[i], A[i])),
        })

    with open("spectral_nd_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved full report to spectral_nd_report.json")


if __name__ == "__main__":
    main()
