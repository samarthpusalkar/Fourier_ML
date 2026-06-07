"""
Quick grid comparison: scalar head + NUFFT vs scalar + split grid
on instrument timbre task. 12 epochs, same data.
"""
import torch
import torch.nn as nn
import numpy as np
from spectral_core import SpectralModel, count_params

SR, MAX_LEN, N_CLASSES = 8000, 8000, 5
INSTRUMENTS = ["piano", "flute", "guitar", "violin", "trumpet"]

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def adsr_envelope(t, a, d, s, r, dur):
    env = np.zeros_like(t)
    ae, de = a, a+d
    rs = dur - r
    env[t <= a] = t[t <= a] / (a + 1e-6)
    mask = (t > a) & (t <= de)
    env[mask] = 1.0 - (1.0 - s) * (t[mask] - a) / (d + 1e-6)
    mask = (t > de) & (t <= rs)
    env[mask] = s
    mask = t > rs
    env[mask] = s * (1.0 - (t[mask] - rs) / (r + 1e-6))
    env[env < 0] = 0
    return env

def generate_tone(inst, fund, dur, sr=SR):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    H, P, A = [], [], []
    if inst == "piano":
        for h in range(1,12): H.append(h*fund*(1+0.0005*h**2)); A.append(1.0/h**0.7); P.append(np.random.uniform(0, 2*np.pi))
        a,d,sus,r = 0.01, 0.3, 0.3, 0.4
    elif inst == "flute":
        for h in [1,2,3]: H.append(h*fund); A.append([1.0,0.15,0.05][h-1]); P.append(np.random.uniform(0, 2*np.pi))
        a,d,sus,r = 0.15, 0.1, 0.6, 0.15
    elif inst == "guitar":
        for h in range(1,10): H.append(h*fund*(1+0.001*h**2)); A.append(1.0/h if h%2==1 else 0.3/h); P.append(np.random.uniform(0, 2*np.pi))
        a,d,sus,r = 0.005, 0.4, 0.2, 0.5
    elif inst == "violin":
        for h in range(1,16): H.append(h*fund); A.append(1.0/h); P.append(0.0)
        a,d,sus,r = 0.1, 0.05, 0.7, 0.15
    else:
        for h in range(1,14): H.append(h*fund); A.append(1.0/h if h%2==1 else 0.2/h); P.append(np.random.uniform(0, 2*np.pi))
        a,d,sus,r = 0.03, 0.1, 0.7, 0.2
    sig = np.zeros_like(t)
    for freq, amp, phase in zip(H, A, P):
        vib = 5.0*np.sin(2*np.pi*5.0*t) if inst=="violin" else 0.0
        sig += amp * np.sin(2*np.pi*freq*t + phase + vib)
    sig *= adsr_envelope(t, a, d, sus, r, dur)
    sig += 0.02*np.random.randn(len(t)) + 0.01*np.sin(2*np.pi*50*t)
    return (sig / (np.max(np.abs(sig))+1e-6)).astype(np.float32)

def make_dataset(n_per_class=400):
    np.random.seed(42)
    data = []
    for cls, inst in enumerate(INSTRUMENTS):
        for _ in range(n_per_class):
            tone = generate_tone(inst, np.random.uniform(100,800), np.random.uniform(0.5,1.0))
            if len(tone) < MAX_LEN: tone = np.pad(tone, (0, MAX_LEN-len(tone)))
            else: tone = tone[:MAX_LEN]
            data.append((tone, cls))
    np.random.shuffle(data)
    n = int(len(data)*0.8)
    return data[:n], data[n:]

def train_eval(model, tr, te, epochs=12, lr=1e-2, wd=1e-4):
    dev = get_device()
    model = model.to(dev)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best = 0.0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        c, t = 0, 0
        with torch.no_grad():
            for xb, yb in te:
                xb, yb = xb.to(dev), yb.to(dev)
                c += (model(xb).argmax(-1) == yb).sum().item()
                t += yb.size(0)
        acc = c / t
        best = max(best, acc)
        if ep <= 3 or ep % 3 == 0 or ep == epochs:
            print(f"  Ep{ep:02d} acc={acc:.4f}")
    return best

def main():
    dev = get_device()
    print(f"Device: {dev}")
    tr, te = make_dataset(400)
    x_tr = torch.stack([torch.from_numpy(d[0]) for d in tr]).unsqueeze(1)
    y_tr = torch.tensor([d[1] for d in tr], dtype=torch.long)
    x_te = torch.stack([torch.from_numpy(d[0]) for d in te]).unsqueeze(1)
    y_te = torch.tensor([d[1] for d in te], dtype=torch.long)
    tr_ld = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, y_tr), 64, shuffle=True)
    te_ld = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_te, y_te), 64, shuffle=False)

    cfgs = [
        ("scalar_split", {"head_type":"scalar", "grid_type":"split"}),
        ("scalar_nufft", {"head_type":"scalar", "grid_type":"nufft"}),
    ]
    results = {}
    for name, kw in cfgs:
        print(f"\n=== {name} ===")
        m = SpectralModel((MAX_LEN,), 1, N_CLASSES, latent_dim=32, num_modes=64,
                          num_mixer_layers=4, activation="square",
                          mixer_dropout=0.1, classifier_dropout=0.3, norm_type="group", **kw)
        print(f"Params: {count_params(m)}")
        results[name] = train_eval(m, tr_ld, te_ld, 12)

    print("\n=== BASELINE MLP ===")
    bl = nn.Sequential(nn.Flatten(), nn.Linear(MAX_LEN,512), nn.ReLU(), nn.Dropout(0.3),
                       nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,N_CLASSES))
    print(f"Params: {count_params(bl)}")
    results["mlp"] = train_eval(bl, tr_ld, te_ld, 12)

    print("\n" + "="*40)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
