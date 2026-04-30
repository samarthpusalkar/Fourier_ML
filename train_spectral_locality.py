"""
Spectral-Locality Network (SL-Net)
====================================
A differentiable hybrid of the Spectral Interpolation Classifier ideas
with standard backpropagation.

Architecture:
  1. Encoder MLP:    high-D input -> low-D latent (learnable locality-preserving projection)
  2. Fourier Layer:  learnable sinusoidal features on the latent space
  3. Classifier:     linear head on Fourier features -> logits

All parameters trained with standard gradient descent.
No numerical integration, no KD-trees, no fixed mapd projections.

Dependencies: torch, numpy  (no torchvision required)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import pickle
import gzip
import urllib.request


# ---------------------------------------------------------------------------
# Minimal MNIST loader without torchvision
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
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class LatentEncoder(nn.Module):
    """
    Replaces hand-coded mapd projection.
    Learns differentiable mapping from input space to low-D latent.
    """
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[256, 128]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FourierFeatureLayer(nn.Module):
    """
    Learnable truncated Fourier series:
        out = dc + sum_n [ A_n cos(w_n * z) + B_n sin(w_n * z) ]
    Each mode has learnable frequency w_n and amplitudes A_n, B_n.
    """
    def __init__(self, latent_dim, num_modes=64):
        super().__init__()
        self.num_modes = num_modes
        self.latent_dim = latent_dim

        self.frequencies = nn.Parameter(torch.randn(num_modes, latent_dim) * 0.1)
        self.A = nn.Parameter(torch.randn(num_modes) * 0.01)
        self.B = nn.Parameter(torch.randn(num_modes) * 0.01)
        self.dc = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        proj = torch.matmul(z, self.frequencies.T)
        cos_feat = torch.cos(proj)
        sin_feat = torch.sin(proj)
        out = self.dc + torch.sum(self.A * cos_feat + self.B * sin_feat, dim=-1, keepdim=True)
        return out


class DistanceWeightedAttention(nn.Module):
    """
    Differentiable replacement for KDTree + manual distance weighting.
    """
    def __init__(self, latent_dim, num_prototypes=64, temperature=1.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, latent_dim) * 0.1)
        self.values = nn.Parameter(torch.randn(num_prototypes, 1) * 0.1)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, z):
        diff = z.unsqueeze(1) - self.prototypes.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        weights = F.softmax(-dist_sq / (self.temperature ** 2 + 1e-6), dim=-1)
        out = torch.matmul(weights, self.values)
        return out


class SpectralLocalityClassifier(nn.Module):
    """
    Full model: encoder -> Fourier features -> classification head.
    """
    def __init__(
        self,
        input_dim,
        num_classes,
        latent_dim=8,
        num_modes=64,
        use_attention=False,
        num_prototypes=64,
        hidden_dims=[256, 128],
    ):
        super().__init__()
        self.encoder = LatentEncoder(input_dim, latent_dim, hidden_dims)
        self.fourier = FourierFeatureLayer(latent_dim, num_modes)
        self.use_attention = use_attention
        if use_attention:
            self.attention = DistanceWeightedAttention(latent_dim, num_prototypes)

        head_in = num_modes + 1 if use_attention else 1
        self.classifier = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        z = self.encoder(x)
        fourier_out = self.fourier(z)
        if self.use_attention:
            attn_out = self.attention(z)
            features = torch.cat([fourier_out, attn_out], dim=-1)
        else:
            features = fourier_out
        logits = self.classifier(features)
        return logits

    def get_fourier_params(self):
        """Return a dict of learned Fourier coefficients/frequencies for inspection."""
        return {
            "frequencies": self.fourier.frequencies.detach().cpu().numpy(),
            "A": self.fourier.A.detach().cpu().numpy(),
            "B": self.fourier.B.detach().cpu().numpy(),
            "dc": self.fourier.dc.detach().cpu().item(),
        }


# ---------------------------------------------------------------------------
# Training / evaluation helpers
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Spectral-Locality Classifier on MNIST")
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--num_modes", type=int, default=64)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--num_prototypes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data/mnist")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load MNIST without torchvision ---
    _download_mnist(args.data_dir)
    x_train, y_train = _load_mnist(args.data_dir, "train")
    x_test, y_test = _load_mnist(args.data_dir, "test")
    input_dim = 28 * 28
    num_classes = 10

    train_loader = torch.utils.data.DataLoader(
        SimpleDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        SimpleDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False
    )

    # --- build model ---
    model = SpectralLocalityClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dim=args.latent_dim,
        num_modes=args.num_modes,
        use_attention=args.use_attention,
        num_prototypes=args.num_prototypes,
        hidden_dims=args.hidden_dims,
    ).to(device)

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt = {
                "state_dict": model.state_dict(),
                "args": vars(args),
                "fourier_params": model.get_fourier_params(),
            }
            torch.save(ckpt, "best_spectral_model.pt")

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    print(f"\nBest test accuracy: {best_acc:.4f}")
    print("Saved checkpoint with Fourier params to best_spectral_model.pt")


if __name__ == "__main__":
    main()
