"""
test_spectral_core.py — Test generic spectral module on diverse data types
==========================================================================
Uses spectral_core.SpectralModel for:
  1. Images     (Fashion-MNIST, 2D)
  2. 1D sequences (synthetic sinusoidal classification)
  3. Tabular     (iris-like synthetic, no spatial dims => rank 0 fallback)

Device priority: MPS > CUDA > CPU
"""
import torch
import torch.nn as nn
import numpy as np
import os, gzip, urllib.request
from spectral_core import SpectralModel, count_params

FASHION_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _download_fashion(data_dir="./data/fashion"):
    os.makedirs(data_dir, exist_ok=True)
    for key, fname in FASHION_FILES.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            urllib.request.urlretrieve(FASHION_URL + fname, fpath)


def _load_fashion(data_dir="./data/fashion", subset="train"):
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


def train_eval(model, train_loader, test_loader, epochs=12, lr=1e-2, wd=1e-4):
    device = get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        if acc > best:
            best = acc
        if epoch <= 3 or epoch % 3 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d} | Test Acc: {acc:.4f}")
    print(f"  Best accuracy: {best:.4f}")
    return best


# ---------------------------------------------------------------------------
# Test 1: Images (Fashion-MNIST, 2D)
# ---------------------------------------------------------------------------
def test_images():
    print("\n" + "=" * 60)
    print("TEST 1: Images — Fashion-MNIST (2D spatial)")
    print("=" * 60)
    _download_fashion()
    x_tr, y_tr = _load_fashion(subset="train")
    x_te, y_te = _load_fashion(subset="test")
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=256, shuffle=False)

    model = SpectralModel(
        spatial_shape=(28, 28), in_channels=1, num_classes=10,
        latent_dim=16, num_mixer_layers=2, activation="square",
        mixer_dropout=0.0, classifier_dropout=0.0, norm_type="batch",
    )
    print(f"  Params: {count_params(model)}")
    return train_eval(model, train_loader, test_loader, epochs=12)


# ---------------------------------------------------------------------------
# Test 2: 1D Sequences (synthetic sinusoidal classification)
# ---------------------------------------------------------------------------
def test_sequences():
    print("\n" + "=" * 60)
    print("TEST 2: 1D Sequences — Synthetic Sinusoidal Classification")
    print("=" * 60)
    np.random.seed(42)
    torch.manual_seed(42)
    n_samples = 5000
    seq_len = 128
    n_classes = 5
    # Each class has a different dominant frequency + phase
    freqs = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    phases = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])

    x = np.zeros((n_samples, 1, seq_len), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        cls = i % n_classes
        t = np.linspace(0, 4 * np.pi, seq_len)
        x[i, 0] = np.sin(freqs[cls] * t + phases[cls]) + 0.1 * np.random.randn(seq_len)
        y[i] = cls

    x_tr, y_tr = torch.from_numpy(x[:4000]), torch.from_numpy(y[:4000])
    x_te, y_te = torch.from_numpy(x[4000:]), torch.from_numpy(y[4000:])
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=64, shuffle=False)

    model = SpectralModel(
        spatial_shape=(seq_len,), in_channels=1, num_classes=n_classes,
        latent_dim=16, num_mixer_layers=2, activation="square",
        mixer_dropout=0.0, classifier_dropout=0.0, norm_type="group",
    )
    print(f"  Params: {count_params(model)}")
    return train_eval(model, train_loader, test_loader, epochs=12)


# ---------------------------------------------------------------------------
# Test 3: Tabular (no spatial dims — rank 0, treated as 1D with length 1)
# ---------------------------------------------------------------------------
def test_tabular():
    print("\n" + "=" * 60)
    print("TEST 3: Tabular — Synthetic Iris-like (4 features, 3 classes)")
    print("=" * 60)
    np.random.seed(42)
    torch.manual_seed(42)
    n_samples = 1500
    n_features = 4
    n_classes = 3

    x = np.random.randn(n_samples, 1, n_features).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        cls = i % n_classes
        x[i, 0, cls] += 2.0  # class-dependent bump
        x[i] += 0.3 * np.random.randn(n_features)
        y[i] = cls

    x_tr, y_tr = torch.from_numpy(x[:1200]), torch.from_numpy(y[:1200])
    x_te, y_te = torch.from_numpy(x[1200:]), torch.from_numpy(y[1200:])
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=32, shuffle=False)

    # Treat tabular as 1D sequence of length 1 — spectral mixer becomes trivial FFT on single point
    # Better: don't use spectral mixers at all for rank 0, just channel projection + classifier
    # spectral_core doesn't support rank 0 natively. Use a tiny 1D workaround.
    # For true tabular: just a linear model. Skip spectral here.
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_features, 64), nn.ReLU(),
        nn.Linear(64, n_classes),
    )
    print(f"  Params: {count_params(model)} (baseline linear — spectral N/A for rank 0)")
    return train_eval(model, train_loader, test_loader, epochs=12)


# ---------------------------------------------------------------------------
# Test 4: 3D Voxels (synthetic small 3D cube)
# ---------------------------------------------------------------------------
def test_voxels():
    print("\n" + "=" * 60)
    print("TEST 4: 3D Voxels — Synthetic 8x8x8 cubes, 4 classes")
    print("=" * 60)
    np.random.seed(42)
    torch.manual_seed(42)
    n_samples = 800
    shape = (8, 8, 8)
    n_classes = 4

    x = np.random.randn(n_samples, 1, *shape).astype(np.float32) * 0.3
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        cls = i % n_classes
        # Each class has a different 3D frequency pattern
        fx, fy, fz = [0.5, 1.0, 1.5, 2.0][cls], [1.0, 0.5, 2.0, 1.5][cls], [1.5, 2.0, 0.5, 1.0][cls]
        z_coords = np.linspace(0, 2 * np.pi, shape[2])
        y_coords = np.linspace(0, 2 * np.pi, shape[1])
        x_coords = np.linspace(0, 2 * np.pi, shape[0])
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        pattern = np.sin(fx * X) * np.cos(fy * Y) * np.sin(fz * Z)
        x[i, 0] += pattern.astype(np.float32)
        y[i] = cls

    x_tr, y_tr = torch.from_numpy(x[:640]), torch.from_numpy(y[:640])
    x_te, y_te = torch.from_numpy(x[640:]), torch.from_numpy(y[640:])
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=32, shuffle=False)

    model = SpectralModel(
        spatial_shape=shape, in_channels=1, num_classes=n_classes,
        latent_dim=8, num_mixer_layers=2, activation="square",
        mixer_dropout=0.0, classifier_dropout=0.0, norm_type="group",
    )
    print(f"  Params: {count_params(model)}")
    return train_eval(model, train_loader, test_loader, epochs=12)


def main():
    device = get_device()
    print(f"Using device: {device}")
    results = {}
    results["images"] = test_images()
    results["sequences"] = test_sequences()
    results["tabular"] = test_tabular()
    results["voxels"] = test_voxels()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, acc in results.items():
        print(f"  {name:12s}: {acc:.4f}")


if __name__ == "__main__":
    main()
