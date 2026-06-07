"""
Quick 7-epoch experiment: Fashion-MNIST as 1D flattened sequence (784,).
Compares Spectral 1D vs Baseline MLP on same flattened input.
LR = 0.01, no cosine scheduler.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import gzip
import urllib.request
from spectral_core import SpectralModel, count_params

FASHION_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_FILES = {
    "train_images": "train-images-idx3-ubyte.gz", "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",   "test_labels": "t10k-labels-idx1-ubyte.gz",
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
    # FLATTEN to 1D: (N, 1, 784)
    images = images.reshape(-1, 1, 784)
    return torch.from_numpy(images), torch.from_numpy(labels)

def train_eval(model, loader_train, loader_test, epochs=7, lr=0.01, wd=1e-4):
    device = get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader_test:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        if acc > best:
            best = acc
        print(f"  Epoch {epoch:02d} | Test Acc: {acc:.4f}")
    print(f"  Best accuracy: {best:.4f}")
    return best

def main():
    device = get_device()
    print(f"Device: {device}")
    _download_fashion()
    x_tr, y_tr = _load_fashion(subset="train")
    x_te, y_te = _load_fashion(subset="test")
    print(f"Data shape: {x_tr.shape} (flattened 1D)")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=256, shuffle=False)

    print("\n" + "=" * 60)
    print("SPECTRAL 1D — Fashion-MNIST flattened to 784")
    print("=" * 60)
    model = SpectralModel(
        spatial_shape=(784,), in_channels=1, num_classes=10,
        latent_dim=32, num_mixer_layers=4, activation="square",
        mixer_dropout=0.0, classifier_dropout=0.0, norm_type="group",
    )
    print(f"Params: {count_params(model)}")
    acc = train_eval(model, train_loader, test_loader, epochs=7, lr=0.01)

    print("\n" + "=" * 60)
    print("BASELINE MLP — same flattened input")
    print("=" * 60)
    baseline = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 10),
    )
    print(f"Params: {count_params(baseline)}")
    acc_b = train_eval(baseline, train_loader, test_loader, epochs=7, lr=0.01)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Spectral 1D: {acc:.4f}")
    print(f"  Baseline MLP: {acc_b:.4f}")

if __name__ == "__main__":
    main()
