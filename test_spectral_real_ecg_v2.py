"""
Real MIT-BIH v2: R-R intervals + Focal Loss + Balanced Batch + Spectral Weights Analysis
==========================================================================================
1. Extract R-R interval (distance to previous R-peak) as scalar feature.
2. Focal Loss: focuses on hard examples, doesn't blanket-weight classes.
3. BalancedBatchSampler: every batch has ~equal class representation.
4. Analyze spectral mixer weight distribution (low vs high freq energy).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wfdb
import os
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from spectral_core import SpectralModel, count_params

DATA_DIR = "./data/mitbih"
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


def extract_beats_with_rr(sig, ann):
    """Extract windows + R-R interval (samples since previous beat)."""
    if sig is None or ann is None:
        return []
    signal = sig.p_signal[:, 0]
    beats = []
    for i, sym in enumerate(ann.symbol):
        if sym not in AAMI_MAP:
            continue
        aami = AAMI_MAP[sym]
        center = ann.sample[i]
        # R-R interval: distance from previous R-peak
        if i > 0:
            rr = center - ann.sample[i - 1]
        else:
            rr = 360  # default ~1 sec
        start = max(0, center - WINDOW // 2)
        end = min(len(signal), start + WINDOW)
        window = signal[start:end]
        if len(window) < WINDOW:
            continue
        window = (window - window.mean()) / (window.std() + 1e-6)
        beats.append((window.astype(np.float32), float(rr), CLASSES.index(aami)))
    return beats


def build_dataset(records):
    all_beats = []
    for rec in records:
        sig, ann = load_record(rec)
        beats = extract_beats_with_rr(sig, ann)
        all_beats.extend(beats)
    windows = np.stack([b[0] for b in all_beats])
    rr = np.array([b[1] for b in all_beats], dtype=np.float32)
    y = np.array([b[2] for b in all_beats], dtype=np.int64)
    return windows, rr, y


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


# ---------------------------------------------------------------------------
# Balanced Batch Sampler
# ---------------------------------------------------------------------------
class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Samples so every batch has roughly equal class counts.
    """
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_classes = len(np.unique(labels))
        self.class_indices = {c: np.where(self.labels == c)[0] for c in range(self.num_classes)}
        self.num_batches = len(labels) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            per_class = self.batch_size // self.num_classes
            for c in range(self.num_classes):
                idx = np.random.choice(self.class_indices[c], size=per_class, replace=True)
                batch.extend(idx)
            # pad to batch_size if not divisible
            while len(batch) < self.batch_size:
                c = np.random.randint(self.num_classes)
                batch.append(np.random.choice(self.class_indices[c]))
            yield batch[:self.batch_size]

    def __len__(self):
        return self.num_batches


# ---------------------------------------------------------------------------
# Model wrapper: spectral + R-R scalar concatenated before classifier
# ---------------------------------------------------------------------------
class SpectralRRModel(nn.Module):
    def __init__(self, spatial_shape, in_channels, num_classes,
                 latent_dim=32, num_modes=64, num_mixer_layers=4,
                 init_scale=2.0, activation="square"):
        super().__init__()
        self.spectral_backbone = SpectralModel(
            spatial_shape=spatial_shape, in_channels=in_channels, num_classes=num_classes,
            latent_dim=latent_dim, num_modes=num_modes, num_mixer_layers=num_mixer_layers,
            init_scale=init_scale, activation=activation,
            mixer_dropout=0.1, classifier_dropout=0.0, norm_type="group",
            head_type="scalar",
        )
        # Replace the classifier in SpectralModel with identity so we can add RR
        self.spectral_backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1 + latent_dim + 1, 128),  # Fourier scalar + latent + RR
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, rr):
        z = self.spectral_backbone.channel_proj(x)
        for mixer in self.spectral_backbone.mixers:
            z = mixer(z)
        z = z.mean(dim=list(range(2, z.ndim)))
        fourier_scalar, z_full = self.spectral_backbone.fourier_head(z)
        # Normalize RR to ~0-1 range
        rr_norm = (rr.unsqueeze(-1) - 360.0) / 360.0
        features = torch.cat([fourier_scalar, z_full, rr_norm], dim=-1)
        return self.head(features)


# ---------------------------------------------------------------------------
# Evaluate with confusion matrix
# ---------------------------------------------------------------------------
def evaluate_detailed(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, rr, yb in loader:
            xb = xb.to(device)
            rr = rr.to(device)
            preds = model(xb, rr).argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))
    f1 = f1_score(all_labels, all_preds, labels=list(range(len(CLASSES))), average=None, zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=CLASSES, labels=list(range(len(CLASSES))), zero_division=0)
    return cm, f1, report


def analyze_spectral_weights(model):
    """Print summary of spectral mixer frequency gains."""
    print("\n" + "=" * 60)
    print("SPECTRAL MIXER WEIGHT ANALYSIS")
    print("=" * 60)
    for i, mixer in enumerate(model.spectral_backbone.mixers):
        gains = mixer.gain_real.detach().cpu().numpy()
        # Average over channels
        avg_gain = gains.mean(axis=0)
        # For 1D, shape is (freq_bins,)
        low = avg_gain[:len(avg_gain)//4].mean()
        mid = avg_gain[len(avg_gain)//4:len(avg_gain)//2].mean()
        high = avg_gain[len(avg_gain)//2:].mean()
        print(f"  Layer {i}: low_freq_gain={low:.4f}, mid_freq_gain={mid:.4f}, high_freq_gain={high:.4f}")


def main():
    device = get_device()
    print(f"Device: {device}")
    print("Loading DS1 (train)...")
    x_tr, rr_tr, y_tr = build_dataset(DS1)
    print("Loading DS2 (test)...")
    x_te, rr_te, y_te = build_dataset(DS2)

    print(f"\nTrain: {len(y_tr)} | Test: {len(y_te)}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: train={np.sum(y_tr == i)}, test={np.sum(y_te == i)}")

    x_tr = torch.from_numpy(x_tr).unsqueeze(1)
    x_te = torch.from_numpy(x_te).unsqueeze(1)
    rr_tr = torch.from_numpy(rr_tr)
    rr_te = torch.from_numpy(rr_te)
    y_tr = torch.from_numpy(y_tr)
    y_te = torch.from_numpy(y_te)

    train_dataset = torch.utils.data.TensorDataset(x_tr, rr_tr, y_tr)
    test_dataset = torch.utils.data.TensorDataset(x_te, rr_te, y_te)

    batch_size = 128
    sampler = BalancedBatchSampler(y_tr.numpy(), batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SpectralRRModel(
        spatial_shape=(WINDOW,), in_channels=1, num_classes=len(CLASSES),
        latent_dim=32, num_mixer_layers=4, activation="square",
    ).to(device)
    print(f"\nParams: {count_params(model)}")

    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(1, 13):
        model.train()
        for xb, rr, yb in train_loader:
            xb, rr, yb = xb.to(device), rr.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb, rr), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, rr, yb in test_loader:
                xb, rr, yb = xb.to(device), rr.to(device), yb.to(device)
                preds = model(xb, rr).argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
        print(f"Epoch {epoch:02d} | Test Acc: {acc:.4f}")

    analyze_spectral_weights(model)

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
