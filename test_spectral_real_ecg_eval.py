"""
Real MIT-BIH with per-class confusion matrix and F1 scores.
Uses existing downloaded data (DS1/DS2 subset already cached).
"""
import torch
import torch.nn as nn
import numpy as np
import wfdb
import os
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from spectral_core import SpectralModel, count_params

DATA_DIR = "./data/mitbih"
BASE_URL = "https://physionet.org/files/mitdb/1.0.0/"

DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 210]

AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}
CLASSES = ['N', 'S', 'V', 'F', 'Q']
WINDOW = 360


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_record(record):
    base = os.path.join(DATA_DIR, str(record))
    try:
        sig = wfdb.rdrecord(base)
        ann = wfdb.rdann(base, 'atr')
        return sig, ann
    except Exception as e:
        print(f"  Load fail {record}: {e}")
        return None, None


def extract_beats(sig, ann):
    if sig is None or ann is None:
        return []
    signal = sig.p_signal[:, 0]
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
        sig, ann = load_record(rec)
        beats = extract_beats(sig, ann)
        all_beats.extend(beats)
    x = np.stack([b[0] for b in all_beats])
    y = np.array([CLASSES.index(b[1]) for b in all_beats], dtype=np.int64)
    return x, y


def evaluate_detailed(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))
    f1 = f1_score(all_labels, all_preds, labels=list(range(len(CLASSES))), average=None)
    report = classification_report(all_labels, all_preds, target_names=CLASSES, labels=list(range(len(CLASSES))))
    return cm, f1, report


def main():
    device = get_device()
    print(f"Device: {device}")
    print("Loading DS1 (train)...")
    x_tr, y_tr = build_dataset(DS1)
    print("Loading DS2 (test)...")
    x_te, y_te = build_dataset(DS2)

    print(f"\nTrain: {len(x_tr)} | Test: {len(x_te)}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: train={np.sum(y_tr == i)}, test={np.sum(y_te == i)}")

    x_tr = torch.from_numpy(x_tr).unsqueeze(1)
    x_te = torch.from_numpy(x_te).unsqueeze(1)
    y_tr = torch.from_numpy(y_tr)
    y_te = torch.from_numpy(y_te)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_te, y_te), batch_size=128, shuffle=False)

    model = SpectralModel(
        spatial_shape=(WINDOW,), in_channels=1, num_classes=len(CLASSES),
        latent_dim=32, num_mixer_layers=4, activation="square",
        mixer_dropout=0.1, classifier_dropout=0.3, norm_type="group",
    ).to(device)
    print(f"\nParams: {count_params(model)}")

    # Inverse-frequency class weights
    class_counts = np.bincount(y_tr.numpy(), minlength=len(CLASSES))
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * len(CLASSES)  # normalize
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class weights: {weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    for epoch in range(1, 13):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch:02d} done")

    print("\n" + "=" * 60)
    print("DETAILED EVALUATION")
    print("=" * 60)
    cm, f1, report = evaluate_detailed(model, test_loader, device)
    print("Confusion Matrix:")
    print("       " + " ".join(f"{c:>5s}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        row = " ".join(f"{cm[i, j]:>5d}" for j in range(len(CLASSES)))
        print(f"  {cls}: {row}")
    print("\nPer-Class F1:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {f1[i]:.4f}")
    print(f"\nMacro F1: {f1.mean():.4f}")
    y_te_np = y_te.numpy()
    print(f"Weighted F1: {np.average(f1, weights=[np.sum(y_te_np == i) for i in range(len(CLASSES))]):.4f}")
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()
