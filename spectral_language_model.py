"""
spectral_language_model.py — Unified Spectral Language Architecture
===================================================================

This model consolidates the findings from previous spectral experiments:
1. SLP / NUFFT: Uses a CoefficientFourierHead with a non-uniform (Kaiser-Bessel) 
   frequency grid. This allows standard gradient descent while predicting mathematically
   interpretable Fourier coefficients.
2. 1D Spatial Mixing: Uses 1D FFT over the sequence length to mix information 
   across the context window (global receptive field).
3. Masked Language Modeling: Uses an MLM objective (like BERT) because FFT operates 
   globally over the sequence.

Architecture:
- Token Embedding -> Latent Dimension
- N layers of 1D Spectral Mixers (FFT -> Gain -> IFFT) over Sequence Length
- Coefficient Fourier Head (projects per-token latent state to Fourier coeffs)
- Linear Classifier (from coeffs to vocabulary logits)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
from collections import Counter
import math

import nltk
from nltk.corpus import brown

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# We will borrow the NUFFT grid generator directly here or from spectral_core
from spectral_core import CoefficientFourierHead

# =============================================================================
# DATA: Masked Language Modeling (MLM) Dataset
# =============================================================================

def load_brown_mlm(vocab_size=4000, seq_length=32, mask_prob=0.15, sentence_limit=30000):
    print("Loading NLTK Brown corpus for MLM...")
    nltk.download('brown', quiet=True)
    raw_sentences = brown.sents()
    if sentence_limit > 0:
        raw_sentences = raw_sentences[:sentence_limit]
        
    print(f"Loaded {len(raw_sentences)} sentences.")
    
    word_counts = Counter()
    processed_sentences = []
    
    for sent in raw_sentences:
        tokens = [w.lower() for w in sent if w.lower().isalpha()]
        if len(tokens) > 2:
            processed_sentences.append(tokens)
            word_counts.update(tokens)
            
    # Build vocabulary + special tokens
    most_common = [w for w, c in word_counts.most_common(vocab_size - 3)]
    vocab = ["<PAD>", "<UNK>", "<MASK>"] + sorted(most_common)
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for i, w in enumerate(vocab)}
    
    PAD_ID = word2id["<PAD>"]
    UNK_ID = word2id["<UNK>"]
    MASK_ID = word2id["<MASK>"]
    
    print(f"Vocabulary Size: {len(vocab)} words.")
    
    # Create fixed length sequences
    sequences = []
    for sent in processed_sentences:
        ids = [word2id.get(w, UNK_ID) for w in sent]
        # Chunk into seq_length
        for i in range(0, len(ids), seq_length):
            chunk = ids[i:i + seq_length]
            if len(chunk) < seq_length:
                chunk = chunk + [PAD_ID] * (seq_length - len(chunk))
            sequences.append(chunk)
            
    # Masking
    inputs = []
    targets = []
    mask_flags = []
    
    np.random.seed(42)
    for seq in sequences:
        inp = list(seq)
        tgt = list(seq)
        mf = [0] * seq_length
        
        for i, token in enumerate(seq):
            if token == PAD_ID:
                tgt[i] = -100  # Ignore index for CrossEntropy
                continue
                
            if np.random.rand() < mask_prob:
                mf[i] = 1
                rand = np.random.rand()
                if rand < 0.8:
                    inp[i] = MASK_ID
                elif rand < 0.9:
                    # Random word
                    inp[i] = np.random.randint(3, len(vocab))
                # 10% keep original
            else:
                tgt[i] = -100 # Only predict masked tokens
                
        inputs.append(inp)
        targets.append(tgt)
        mask_flags.append(mf)
        
    print(f"Generated {len(inputs)} sequences of length {seq_length}.")
    
    return (torch.tensor(inputs, dtype=torch.long), 
            torch.tensor(targets, dtype=torch.long),
            vocab, word2id, id2word)

# =============================================================================
# ARCHITECTURE: 1D Spectral Mixer
# =============================================================================

class SpectralMixer1D(nn.Module):
    """
    1D FFT over the sequence length.
    Acts as a global sequence mixer (replaces self-attention).
    """
    def __init__(self, latent_dim, seq_length, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        
        self.freq_shape = seq_length // 2 + 1
        
        # Learnable complex gain per frequency bin and per channel
        self.gain_real = nn.Parameter(torch.ones(latent_dim, self.freq_shape) * 0.5)
        self.gain_imag = nn.Parameter(torch.zeros(latent_dim, self.freq_shape))
        
        self.norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (Batch, Seq_Len, Channels)
        # We need to take FFT over Seq_Len, so dim=1
        
        x_fft = torch.fft.rfft(x, dim=1) # (B, Freq, C)
        
        # Gain: (C, Freq) -> Permute to (Freq, C) for broadcasting
        gain = torch.view_as_complex(torch.stack([self.gain_real, self.gain_imag], dim=-1))
        gain = gain.permute(1, 0) # (Freq, C)
        
        x_filtered = x_fft * gain.unsqueeze(0) # (B, Freq, C) * (1, Freq, C)
        
        x_out = torch.fft.irfft(x_filtered, n=self.seq_length, dim=1)
        
        x_out = self.activation(x_out)
        x_out = self.dropout(x_out)
        
        return self.norm(x_out + x)

# =============================================================================
# ARCHITECTURE: Spectral Language Model
# =============================================================================

class SpectralLanguageModel(nn.Module):
    def __init__(self, vocab_size, seq_length, latent_dim=128, num_modes=64, 
                 num_mixer_layers=4, init_scale=2.0, head_hidden=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        
        self.embedding = nn.Embedding(vocab_size, latent_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(seq_length, latent_dim)
        
        self.mixers = nn.ModuleList([
            SpectralMixer1D(latent_dim, seq_length, dropout)
            for _ in range(num_mixer_layers)
        ])
        
        # NUFFT Coefficient Head maps from Latent -> Fourier Coefficients [a0, an, bn]
        self.fourier_head = CoefficientFourierHead(latent_dim, num_modes, init_scale, grid_type="nufft")
        
        coeff_dim = 1 + 2 * num_modes
        
        # From Coefficients to Vocabulary Logits
        self.classifier = nn.Sequential(
            nn.Linear(coeff_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, vocab_size)
        )
        
    def forward(self, x):
        # x: (B, Seq_Len)
        B, L = x.shape
        
        # 1. Embed
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        z = self.embedding(x) + self.pos_embedding(pos) # (B, L, C)
        
        # 2. Global Sequence Mixing via FFT
        for mixer in self.mixers:
            z = mixer(z)
            
        # 3. Project to Fourier Space
        # z: (B, L, C) -> reshape to (B*L, C)
        z_flat = z.view(B * L, self.latent_dim)
        
        coeffs, _ = self.fourier_head(z_flat) # (B*L, coeff_dim)
        
        # 4. Predict Logits
        logits_flat = self.classifier(coeffs) # (B*L, V)
        logits = logits_flat.view(B, L, self.vocab_size) # (B, L, V)
        
        return logits, coeffs

# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=int, default=64, help="Fourier modes in NUFFT Head")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of 1D FFT mixers")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seq_length", type=int, default=32, help="Sequence length")
    parser.add_argument("--vocab_size", type=int, default=4000, help="Vocabulary size")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using Device: {device}")
    
    inputs, targets, vocab, word2id, id2word = load_brown_mlm(
        vocab_size=args.vocab_size, seq_length=args.seq_length
    )
    
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = SpectralLanguageModel(
        vocab_size=len(vocab), 
        seq_length=args.seq_length,
        latent_dim=args.latent_dim,
        num_modes=args.modes,
        num_mixer_layers=args.layers
    ).to(device)
    
    print("\n" + "="*60)
    print("UNIFIED SPECTRAL LANGUAGE MODEL (MLM)")
    print("="*60)
    print(f"Vocab: {len(vocab)} | Seq Len: {args.seq_length} | Latent: {args.latent_dim}")
    print(f"Mixers: {args.layers} Layers | NUFFT Modes: {args.modes}")
    print(f"Total Parameters: {count_params(model):,}")
    print("="*60 + "\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100) # Only compute loss on masked tokens
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_masked = 0
        
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            logits, coeffs = model(xb) # (B, L, V)
            
            # Flatten to compute loss
            logits_flat = logits.view(-1, len(vocab))
            yb_flat = yb.view(-1)
            
            loss = criterion(logits_flat, yb_flat)
            
            # Add spectral regularization to coefficients
            a_n = coeffs[:, 1:1+args.modes]
            b_n = coeffs[:, 1+args.modes:]
            # Simple L2 on the high-frequency components
            reg_loss = 1e-4 * (a_n**2 + b_n**2).mean()
            
            (loss + reg_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy metric
            mask = yb_flat != -100
            if mask.sum() > 0:
                preds = logits_flat.argmax(dim=-1)
                total_correct += (preds[mask] == yb_flat[mask]).sum().item()
                total_masked += mask.sum().item()
                
        scheduler.step()
        
        acc = (total_correct / total_masked) * 100 if total_masked > 0 else 0
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {total_loss/len(loader):.4f} | MLM Acc: {acc:.2f}% | Scale: {model.fourier_head.scale.item():.4f}")

    # Generate Topology Plot at the end
    print("Generating topological plot...")
    model.eval()
    
    # Get POS tags
    print("Assigning POS tags to vocabulary...")
    tagged = nltk.pos_tag(vocab)
    pos_map = {}
    for word, tag in tagged:
        if tag.startswith('NN'): pos_map[word] = 'Noun'
        elif tag.startswith('VB'): pos_map[word] = 'Verb'
        elif tag.startswith('JJ'): pos_map[word] = 'Adjective'
        else: pos_map[word] = 'Other'
        
    all_ids = torch.arange(len(vocab), device=device)
    with torch.no_grad():
        z = model.embedding(all_ids)
        coeffs, _ = model.fourier_head(z)
        coeffs_np = coeffs.cpu().numpy()
        
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    coords = tsne.fit_transform(coeffs_np)
    
    print("Generating topological plot...")
    plt.figure(figsize=(12, 10), facecolor='#f8f9fa')
    
    color_scheme = {
        'Noun': '#1a73e8',       # Blue
        'Verb': '#d93025',       # Red
        'Adjective': '#137333',  # Green
        'Other': '#bdc3c7'       # Gray
    }
    
    for cat in ['Other', 'Noun', 'Verb', 'Adjective']:
        indices = [i for i, w in enumerate(vocab) if pos_map.get(w, 'Other') == cat]
        if not indices: continue
        
        x = coords[indices, 0]
        y = coords[indices, 1]
        
        alpha = 0.2 if cat == 'Other' else 0.7
        size = 15 if cat == 'Other' else 35
        zorder = 1 if cat == 'Other' else 2
        
        plt.scatter(x, y, c=color_scheme[cat], label=cat, alpha=alpha, s=size, zorder=zorder, edgecolors='none')
        
    np.random.seed(42)
    for cat in ['Noun', 'Verb', 'Adjective']:
        cat_indices = [i for i, w in enumerate(vocab) if pos_map.get(w, 'Other') == cat]
        if not cat_indices: continue
        
        sample_indices = np.random.choice(cat_indices, min(10, len(cat_indices)), replace=False)
        for idx in sample_indices:
            word = vocab[idx]
            plt.annotate(word, (coords[idx, 0], coords[idx, 1]), 
                         fontsize=9, alpha=0.9,
                         xytext=(3, 3), textcoords='offset points')
                         
    plt.title("Unified Model: Topological Emergence of Grammar in Fourier Space", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(frameon=True, facecolor='white', edgecolor='gray', fontsize=12, markerscale=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/unified_topology.png", dpi=300, bbox_inches='tight')
    print("Saved topological plot to results/unified_topology.png")
    plt.close()

if __name__ == "__main__":
    main()
