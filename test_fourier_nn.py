"""
test_fourier_nn.py — Compare Fourier-NN vs Standard MLP
=======================================================
Datasets:
  1. Fashion-MNIST (flattened 784) — vector input
  2. Realistic instrument timbre (8000 samples) — vector input
  3. Synthetic sinusoidal classification — vector input

Models:
  A. FourierNN    : FourierLinear layers
  B. Standard MLP : ReLU layers, same widths
"""
import torch
import torch.nn as nn
import numpy as np
import os, gzip, urllib.request
from spectral_fnn import FourierMLP, count_params as fnn_count

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


def _download_fashion():
    os.makedirs("./data/fashion", exist_ok=True)
    for key, fname in FASHION_FILES.items():
        fpath = os.path.join("./data/fashion", fname)
        if not os.path.exists(fpath):
            urllib.request.urlretrieve(FASHION_URL + fname, fpath)

def _load_fashion(subset="train"):
    prefix = "train" if subset == "train" else "t10k"
    ip = os.path.join("./data/fashion", f"{prefix}-images-idx3-ubyte.gz")
    lp = os.path.join("./data/fashion", f"{prefix}-labels-idx1-ubyte.gz")
    with gzip.open(ip, "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28).astype(np.float32) / 255.0
    with gzip.open(lp, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return torch.from_numpy(images), torch.from_numpy(labels)


def train_eval(model, tr, te, epochs=12, lr=1e-2, wd=1e-4):
    device = get_device()
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(); criterion(model(xb), yb).backward(); optimizer.step()
        model.eval()
        c, t = 0, 0
        with torch.no_grad():
            for xb, yb in te:
                xb, yb = xb.to(device), yb.to(device)
                c += (model(xb).argmax(-1) == yb).sum().item()
                t += yb.size(0)
        acc = c / t; best = max(best, acc)
        if epoch <= 3 or epoch % 3 == 0 or epoch == epochs:
            print(f"  Ep{epoch:02d} acc={acc:.4f}")
    print(f"  Best: {best:.4f}")
    return best


# ---------------------------------------------------------------------------
# Test 1: Fashion-MNIST (flattened 784)
# ---------------------------------------------------------------------------
def test_fashion():
    print(f"\n{'='*60}\nTEST 1: Fashion-MNIST flattened (784)\n{'='*60}")
    _download_fashion()
    x_tr, y_tr = _load_fashion("train")
    x_te, y_te = _load_fashion("test")
    tr = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, y_tr), 256, shuffle=True)
    te = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_te, y_te), 256, shuffle=False)

    # FourierNN [256, 128, 64] with 32 modes
    print("\n--- FourierNN [256→128→64]  modes=32 ---")
    fnn = FourierMLP(784, [256, 128, 64], 10, num_modes=32, dropout=0.3, init_scale=2.0)
    print(f"  Params: {fnn_count(fnn):,}")
    acc_fnn = train_eval(fnn, tr, te, 12)

    print("\n--- Standard MLP [256→128→64] ---")
    mlp = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(64, 10))
    print(f"  Params: {fnn_count(mlp):,}")
    acc_mlp = train_eval(mlp, tr, te, 12)
    return acc_fnn, acc_mlp


# ---------------------------------------------------------------------------
# Test 2: Synthetic sinusoidal classification
# ---------------------------------------------------------------------------
def test_sinusoid():
    print(f"\n{'='*60}\nTEST 2: Synthetic sinusoidal (128 dims, 5 classes)\n{'='*60}")
    np.random.seed(42); torch.manual_seed(42)
    n, L, C = 5000, 128, 5
    freqs = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    phases = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    x = np.zeros((n, L), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cls = i % C
        t = np.linspace(0, 4*np.pi, L)
        x[i] = np.sin(freqs[cls]*t + phases[cls]) + 0.1*np.random.randn(L)
        y[i] = cls
    x_tr, y_tr = torch.from_numpy(x[:4000]), torch.from_numpy(y[:4000])
    x_te, y_te = torch.from_numpy(x[4000:]), torch.from_numpy(y[4000:])
    tr = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, y_tr), 64, shuffle=True)
    te = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_te, y_te), 64, shuffle=False)

    print("\n--- FourierNN [128→64] modes=16 ---")
    fnn = FourierMLP(128, [128, 64], C, num_modes=16, dropout=0.1, init_scale=2.0)
    print(f"  Params: {fnn_count(fnn):,}")
    acc_fnn = train_eval(fnn, tr, te, 12)

    print("\n--- Standard MLP [128→64] ---")
    mlp = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(64, C))
    print(f"  Params: {fnn_count(mlp):,}")
    acc_mlp = train_eval(mlp, tr, te, 12)
    return acc_fnn, acc_mlp


# ---------------------------------------------------------------------------
# Test 3: Realistic audio timbre (matching existing test)
# ---------------------------------------------------------------------------
SR, MAX_LEN = 8000, 8000
INSTS = ["piano", "flute", "guitar", "violin", "trumpet"]

def gen_tone(inst, fund, dur):
    t = np.linspace(0, dur, int(SR*dur), endpoint=False)
    H, P, A = [], [], []
    if inst == "piano":
        for h in range(1,12): H.append(h*fund*(1+0.0005*h**2)); A.append(1/h**0.7); P.append(np.random.uniform(0, 2*np.pi))
        a,d,s,r = 0.01,0.3,0.3,0.4
    elif inst == "flute":
        for h in [1,2,3]: H.append(h*fund); A.append([1.0,0.15,0.05][h-1]); P.append(np.random.uniform(0, 2*np.pi))
        a,d,s,r = 0.15,0.1,0.6,0.15
    elif inst == "guitar":
        for h in range(1,10): H.append(h*fund*(1+0.001*h**2)); A.append(1/h if h%2==1 else 0.3/h); P.append(np.random.uniform(0, 2*np.pi))
        a,d,s,r = 0.005,0.4,0.2,0.5
    elif inst == "violin":
        for h in range(1,16): H.append(h*fund); A.append(1/h); P.append(0.0)
        a,d,s,r = 0.1,0.05,0.7,0.15
    else:
        for h in range(1,14): H.append(h*fund); A.append(1/h if h%2==1 else 0.2/h); P.append(np.random.uniform(0, 2*np.pi))
        a,d,s,r = 0.03,0.1,0.7,0.2
    sig = np.zeros_like(t)
    for freq, amp, ph in zip(H, A, P):
        vib = 5.0*np.sin(2*np.pi*5.0*t) if inst=="violin" else 0.0
        sig += amp*np.sin(2*np.pi*freq*t + ph + vib)
    def adsr_env(t, attack, decay, sustain, release, duration):
        env = np.zeros_like(t); ae, de = attack, attack+decay; rs = duration - release
        env[t <= ae] = t[t <= ae]/(attack+1e-6)
        mask = (t > ae) & (t <= de); env[mask] = 1 - (1 - sustain)*(t[mask]-ae)/(decay+1e-6)
        mask = (t > de) & (t <= rs); env[mask] = sustain
        mask = t > rs; env[mask] = sustain * (1 - (t[mask]-rs)/(release+1e-6))
        env[env < 0] = 0; return env
    sig *= adsr_env(t, a, d, s, r, dur)
    sig += 0.02*np.random.randn(len(t)) + 0.01*np.sin(2*np.pi*50*t)
    return (sig / (np.max(np.abs(sig))+1e-6)).astype(np.float32)

def make_audio(n_per=400):
    np.random.seed(42)
    data = []
    for cls, inst in enumerate(INSTS):
        for _ in range(n_per):
            tone = gen_tone(inst, np.random.uniform(100,800), np.random.uniform(0.5,1.0))
            if len(tone) < MAX_LEN: tone = np.pad(tone, (0, MAX_LEN-len(tone)))
            else: tone = tone[:MAX_LEN]
            data.append((tone, cls))
    np.random.shuffle(data); n = int(len(data)*0.8)
    return data[:n], data[n:]

def test_audio():
    print(f"\n{'='*60}\nTEST 3: Realistic audio timbre (8000)\n{'='*60}")
    tr_d, te_d = make_audio(400)
    x_tr = torch.stack([torch.from_numpy(d[0]) for d in tr_d])
    y_tr = torch.tensor([d[1] for d in tr_d], dtype=torch.long)
    x_te = torch.stack([torch.from_numpy(d[0]) for d in te_d])
    y_te = torch.tensor([d[1] for d in te_d], dtype=torch.long)
    tr = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, y_tr), 64, shuffle=True)
    te = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_te, y_te), 64, shuffle=False)

    print("\n--- FourierNN [512→256] modes=16 ---")
    fnn = FourierMLP(MAX_LEN, [512, 256], 5, num_modes=16, dropout=0.3, init_scale=2.0)
    print(f"  Params: {fnn_count(fnn):,}")
    acc_fnn = train_eval(fnn, tr, te, 12)

    print("\n--- Standard MLP [512→256] ---")
    mlp = nn.Sequential(nn.Linear(MAX_LEN, 512), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(256, 5))
    print(f"  Params: {fnn_count(mlp):,}")
    acc_mlp = train_eval(mlp, tr, te, 12)
    return acc_fnn, acc_mlp


def main():
    print(f"Device: {get_device()}")
    r1 = test_fashion()
    r2 = test_sinusoid()
    r3 = test_audio()

    print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
    print(f"Fashion-MNIST:    FourierNN={r1[0]:.4f}  MLP={r1[1]:.4f}")
    print(f"Sinusoidal:       FourierNN={r2[0]:.4f}  MLP={r2[1]:.4f}")
    print(f"Audio Timbre:     FourierNN={r3[0]:.4f}  MLP={r3[1]:.4f}")


if __name__ == "__main__":
    main()
