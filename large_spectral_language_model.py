"""
large_spectral_language_model.py — Large Scale Spectral Language Model
======================================================================
Integrates the architectural convergence of the v8 branch:
1. AdaptiveFourierHead: Data-adaptive learnable frequency spacing.
2. GenericSpectralMixer + LearnableSquareWave: High capacity spectral mixing.
3. PolyakAdamW: Optimal step-size damping for rapid convergence.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import nltk
from nltk.corpus import brown
from spectral_core import CoefficientFourierHead
# =============================================================================
# V8 ARCHITECTURE COMPONENTS
# =============================================================================

class LearnableSquareWave(nn.Module):
    """y = A * tanh(steepness * sin(freq * x + phase))"""
    def __init__(self, steepness=10.0):
        super().__init__()
        self.amplitude = nn.Parameter(torch.tensor(1.0))
        self.log_freq  = nn.Parameter(torch.tensor(0.0))
        self.phase     = nn.Parameter(torch.tensor(0.0))
        self.steepness = steepness

    def forward(self, x):
        # Clamp log_freq to prevent exp() from exploding
        freq = torch.exp(torch.clamp(self.log_freq, max=10.0))
        return self.amplitude * torch.tanh(self.steepness * torch.sin(freq * x + self.phase))


class GenericSpectralMixer1D(nn.Module):
    """1D Sequence Mixer using generic FFT structure."""
    def __init__(self, channels, seq_length):
        super().__init__()
        self.channels = channels
        self.spatial_shape = (seq_length,)
        
        # rfft on dim=1
        self.freq_shape = (seq_length // 2 + 1,)
        
        self.gain_real = nn.Parameter(torch.ones(channels, *self.freq_shape) * 0.5)
        self.gain_imag = nn.Parameter(torch.zeros(channels, *self.freq_shape))
        self.norm = nn.LayerNorm(channels)
        self.activation = LearnableSquareWave()

    def forward(self, x):
        # x: (B, L, C)
        # torch.fft.rfftn expects spatial dims at the end normally, but we can do dim=1
        x_fft = torch.fft.rfft(x, dim=1) # (B, Freq, C)
        
        gain = torch.view_as_complex(torch.stack([self.gain_real, self.gain_imag], dim=-1))
        # gain: (C, Freq) -> (Freq, C)
        gain = gain.permute(1, 0)
        
        x_filtered = x_fft * gain.unsqueeze(0)
        x_out = torch.fft.irfft(x_filtered, n=self.spatial_shape[0], dim=1)
        
        x_out = self.activation(x_out)
        return self.norm(x_out + x)





# =============================================================================
# DATA AND MODEL
# =============================================================================

def load_brown_mlm(vocab_size=4000, seq_length=64, mask_prob=0.15, sentence_limit=40000):
    print("Loading NLTK Brown corpus for MLM...")
    nltk.download('brown', quiet=True)
    raw_sentences = brown.sents()
    if sentence_limit > 0:
        raw_sentences = raw_sentences[:sentence_limit]
        
    word_counts = Counter()
    processed_sentences = []
    
    for sent in raw_sentences:
        tokens = [w.lower() for w in sent if w.lower().isalpha()]
        if len(tokens) > 2:
            processed_sentences.append(tokens)
            word_counts.update(tokens)
            
    most_common = [w for w, c in word_counts.most_common(vocab_size - 3)]
    vocab = ["<PAD>", "<UNK>", "<MASK>"] + sorted(most_common)
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for i, w in enumerate(vocab)}
    
    PAD_ID, UNK_ID, MASK_ID = word2id["<PAD>"], word2id["<UNK>"], word2id["<MASK>"]
    
    sequences = []
    for sent in processed_sentences:
        ids = [word2id.get(w, UNK_ID) for w in sent]
        for i in range(0, len(ids), seq_length):
            chunk = ids[i:i + seq_length]
            if len(chunk) < seq_length:
                chunk = chunk + [PAD_ID] * (seq_length - len(chunk))
            sequences.append(chunk)
            
    inputs, targets = [], []
    np.random.seed(42)
    for seq in sequences:
        inp, tgt = list(seq), list(seq)
        for i, token in enumerate(seq):
            if token == PAD_ID:
                tgt[i] = -100
                continue
            if np.random.rand() < mask_prob:
                rand = np.random.rand()
                if rand < 0.8: inp[i] = MASK_ID
                elif rand < 0.9: inp[i] = np.random.randint(3, len(vocab))
            else:
                tgt[i] = -100
        inputs.append(inp)
        targets.append(tgt)
        
    return (torch.tensor(inputs, dtype=torch.long), 
            torch.tensor(targets, dtype=torch.long),
            vocab, word2id, id2word)


class LargeSpectralLM(nn.Module):
    def __init__(self, vocab_size, seq_length=64, latent_dim=256, num_modes=128, 
                 num_layers=6, head_hidden=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        
        self.embedding = nn.Embedding(vocab_size, latent_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(seq_length, latent_dim)
        
        self.mixers = nn.ModuleList([
            GenericSpectralMixer1D(latent_dim, seq_length)
            for _ in range(num_layers)
        ])
        
        self.fourier_head = CoefficientFourierHead(latent_dim, num_modes, init_scale=2.0, grid_type="nufft")
        coeff_dim = 1 + 2 * num_modes
        
        self.classifier = nn.Sequential(
            nn.Linear(coeff_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, vocab_size)
        )
        
    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        z = self.embedding(x) + self.pos_embedding(pos)
        
        for mixer in self.mixers:
            z = mixer(z)
            
        z_flat = z.view(B * L, self.latent_dim)
        coeffs, _ = self.fourier_head(z_flat)
        logits_flat = self.classifier(coeffs)
        
        return logits_flat.view(B, L, self.vocab_size), coeffs

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3, help="Base LR for AdamW")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_modes", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    inputs, targets, vocab, word2id, id2word = load_brown_mlm(vocab_size=4000, seq_length=64)
    
    # Train/Test Split (90/10)
    dataset_size = len(inputs)
    split_idx = int(dataset_size * 0.9)
    train_inputs, train_targets = inputs[:split_idx], targets[:split_idx]
    test_inputs, test_targets = inputs[split_idx:], targets[split_idx:]
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_inputs, train_targets), batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_inputs, test_targets), batch_size=args.batch_size, shuffle=False
    )
    
    model = LargeSpectralLM(
        vocab_size=len(vocab), 
        latent_dim=args.latent_dim, 
        num_modes=args.num_modes, 
        num_layers=args.num_layers
    ).to(device)
    
    print("\n" + "="*60)
    print("LARGE SPECTRAL LM (v8 Architecture + AdamW)")
    print("="*60)
    print(f"Total Parameters: {count_params(model):,}")
    print(f"Optimizer: AdamW | LR: {args.lr}")
    print("="*60 + "\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_loss = 999.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total_masked = 0.0, 0, 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            logits, coeffs = model(xb)
            
            logits_flat = logits.view(-1, len(vocab))
            yb_flat = yb.view(-1)
            
            loss = criterion(logits_flat, yb_flat)
            
            # Spectral regularization
            a_n = coeffs[:, 1:1+args.num_modes]
            b_n = coeffs[:, 1+args.num_modes:]
            reg_loss = 1e-4 * (a_n**2 + b_n**2).mean()
            
            (loss + reg_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            mask = yb_flat != -100
            if mask.sum() > 0:
                preds = logits_flat.argmax(dim=-1)
                total_correct += (preds[mask] == yb_flat[mask]).sum().item()
                total_masked += mask.sum().item()
                
        avg_loss = total_loss / len(train_loader)
        acc = (total_correct / total_masked) * 100 if total_masked > 0 else 0
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")
        
        if epoch % 7 == 0 or epoch == args.epochs:
            model.eval()
            test_loss, test_correct, test_masked = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits, coeffs = model(xb)
                    logits_flat = logits.view(-1, len(vocab))
                    yb_flat = yb.view(-1)
                    loss = criterion(logits_flat, yb_flat)
                    test_loss += loss.item()
                    mask = yb_flat != -100
                    if mask.sum() > 0:
                        preds = logits_flat.argmax(dim=-1)
                        test_correct += (preds[mask] == yb_flat[mask]).sum().item()
                        test_masked += mask.sum().item()
            t_loss = test_loss / len(test_loader)
            t_acc = (test_correct / test_masked) * 100 if test_masked > 0 else 0
            print(f"    -> [Test] Loss: {t_loss:.4f} | Acc: {t_acc:.2f}%")
            
            if t_loss < best_loss:
                best_loss = t_loss
                os.makedirs("results", exist_ok=True)
                torch.save(model.state_dict(), "results/best_large_spectral.pt")

    print(f"\nFinal Best Test Loss: {best_loss:.4f}")
    print(f"Saved best model to results/best_large_spectral.pt")
    
    # Topology evaluation
    print("\nGenerating topological plot...")
    model.eval()
    nltk.download('averaged_perceptron_tagger', quiet=True)
    tagged = nltk.pos_tag(vocab)
    pos_map = {}
    for w, t in tagged:
        if t.startswith('NN'): pos_map[w] = 'Noun'
        elif t.startswith('VB'): pos_map[w] = 'Verb'
        elif t.startswith('JJ'): pos_map[w] = 'Adjective'
        else: pos_map[w] = 'Other'
        
    all_ids = torch.arange(len(vocab), device=device)
    with torch.no_grad():
        z = model.embedding(all_ids)
        coeffs, _ = model.fourier_head(z)
        coeffs = coeffs.cpu().numpy()
        
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    coords = tsne.fit_transform(coeffs)
    
    plt.figure(figsize=(12, 10), facecolor='#f8f9fa')
    color_scheme = {'Noun': '#1a73e8', 'Verb': '#d93025', 'Adjective': '#137333', 'Other': '#bdc3c7'}
    
    for cat in ['Other', 'Noun', 'Verb', 'Adjective']:
        indices = [i for i, w in enumerate(vocab) if pos_map.get(w, 'Other') == cat]
        if not indices: continue
        alpha, size, zorder = (0.2, 15, 1) if cat == 'Other' else (0.7, 35, 2)
        plt.scatter(coords[indices, 0], coords[indices, 1], c=color_scheme[cat], label=cat, 
                    alpha=alpha, s=size, zorder=zorder, edgecolors='none')
        
    np.random.seed(42)
    for cat in ['Noun', 'Verb', 'Adjective']:
        cat_indices = [i for i, w in enumerate(vocab) if pos_map.get(w, 'Other') == cat]
        if not cat_indices: continue
        sample_indices = np.random.choice(cat_indices, min(10, len(cat_indices)), replace=False)
        for idx in sample_indices:
            plt.annotate(vocab[idx], (coords[idx, 0], coords[idx, 1]), fontsize=9, alpha=0.9, xytext=(3, 3), textcoords='offset points')
                         
    plt.title("Large V8 Model + Polyak: Topological Grammar in Fourier Space", fontsize=16, fontweight='bold', pad=15)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/large_v8_topology.png", dpi=300, bbox_inches='tight')
    print("Saved plot to results/large_v8_topology.png")

if __name__ == "__main__":
    main()
