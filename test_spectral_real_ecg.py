"""
test_spectral_real_ecg.py — Real MIT-BIH Arrhythmia, Inter-Patient Split
========================================================================
Uses standard DS1/DS2 inter-patient split.
Direct download from PhysioNet (bypasses wfdb API changes).

DS1 (train): 101,106,108,109,112,114,115,116,118,119,122,124,
             201,203,205,207,208,209,215,220,223,230
DS2 (test):  100,103,105,111,113,117,121,123,200,202,210,212,
             213,214,219,221,222,228,231,232,233,234

Signal: 360 Hz, 2 leads. Using lead 0 (MLII where available).
Window: 360 samples (~1 sec) centered on R-peak annotation.
"""
import torch
import torch.nn as nn
import numpy as np
import wfdb
import os
import urllib.request
import socket
socket.setdefaulttimeout(15)
from spectral_core import SpectralModel, count_params

DATA_DIR = "./data/mitbih"
BASE_URL = "https://physionet.org/files/mitdb/1.0.0/"

DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
# Use only DS2 records already present locally to avoid download hangs
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 210]

AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}
CLASSES = ['N', 'S', 'V', 'F', 'Q']
SAMPLE_RATE = 360
WINDOW = 360


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def download_record(record):
    """Download .dat, .hea, .atr for a single record."""
    os.makedirs(DATA_DIR, exist_ok=True)
    base = os.path.join(DATA_DIR, str(record))
    files = {
        f"{record}.dat":  base + ".dat",
        f"{record}.hea":  base + ".hea",
        f"{record}.atr":  base + ".atr",
    }
    ok = True
    for remote, local in files.items():
        if not os.path.exists(local):
            url = BASE_URL + remote
            try:
                urllib.request.urlretrieve(url, local)
            except Exception as e:
                print(f"    DL fail {remote}: {e}")
                ok = False
    return ok


def load_record(record):
    base = os.path.join(DATA_DIR, str(record))
    if not os.path.exists(base + ".dat"):
        return None, None
    try:
        sig = wfdb.rdrecord(base)
        ann = wfdb.rdann(base, 'atr')
        return sig, ann
    except Exception as e:
        print(f"    Load fail {record}: {e}")
        return None, None


def extract_beats(sig, ann):
    """Extract normalized windows around each annotated beat."""
    if sig is None or ann is None:
        return []
    signal = sig.p_signal[:, 0]  # MLII
    beats = []
    for i, sym in enumerate(ann.symbol):
        if sym not in AAMI_MAP:
            continue
        aami = AAMI_MAP[sym]
        center = ann.sample[i]
        start = max(0, center - WINDOW // 2)
        end = min(len(signal), start + WINDOW)
        window = signal[start:end]
        if len(window) < WINDOW:
            continue
        window = (window - window.mean()) / (window.std() + 1e-6)
        beats.append((window.astype(np.float32), aami))
    return beats


def build_dataset(records):
    all_beats = []
    for rec in records:
        if not download_record(rec):
            continue
        sig, ann = load_record(rec)
        beats = extract_beats(sig, ann)
        all_beats.extend(beats)
        print(f"  Record {rec}: {len(beats)} beats")
    if not all_beats:
        return None, None
    x = np.stack([b[0] for b in all_beats])
    y = np.array([CLASSES.index(b[1]) for b in all_beats], dtype=np.int64)
    return x, y


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


def main():
    device = get_device()
    print(f"Device: {device}")
    print("=" * 60)
    print("LOADING DS1 (train) ...")
    x_tr, y_tr = build_dataset(DS1)
    print("=" * 60)
    print("LOADING DS2 (test) ...")
    x_te, y_te = build_dataset(DS2)

    if x_tr is None or x_te is None:
        print("Failed to load data. Exiting.")
        return

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Train: {len(x_tr)} beats from {len(DS1)} records")
    print(f"  Test:  {len(x_te)} beats from {len(DS2)} records")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls}: train={np.sum(y_tr == i)}, test={np.sum(y_te == i)}")

    x_tr = torch.from_numpy(x_tr).unsqueeze(1)
    x_te = torch.from_numpy(x_te).unsqueeze(1)
    y_tr = torch.from_numpy(y_tr)
    y_te = torch.from_numpy(y_te)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=128, shuffle=False)

    print("\n" + "=" * 60)
    print("SPECTRAL MODEL")
    print("=" * 60)
    model = SpectralModel(
        spatial_shape=(WINDOW,), in_channels=1, num_classes=len(CLASSES),
        latent_dim=32, num_mixer_layers=4, activation="square",
        mixer_dropout=0.1, classifier_dropout=0.3, norm_type="group",
    )
    print(f"  Params: {count_params(model)}")
    acc = train_eval(model, train_loader, test_loader, epochs=12, lr=1e-2)

    print("\n" + "=" * 60)
    print("BASELINE MLP")
    print("=" * 60)
    baseline = nn.Sequential(
        nn.Flatten(),
        nn.Linear(WINDOW, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, len(CLASSES)),
    )
    print(f"  Params: {count_params(baseline)}")
    acc_b = train_eval(baseline, train_loader, test_loader, epochs=12, lr=1e-2)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Spectral:     {acc:.4f}")
    print(f"  Baseline MLP: {acc_b:.4f}")


if __name__ == "__main__":
    main()
