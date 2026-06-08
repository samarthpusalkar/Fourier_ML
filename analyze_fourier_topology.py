"""
analyze_fourier_topology.py — Topological Emergence of Syntax and Semantics in Fourier Space
============================================================================================
1. Trains Fourier word embeddings from raw text on the Brown corpus.
2. Extracts the raw learned Fourier coefficients for the vocabulary.
3. Automatically POS tags the vocabulary (Nouns, Verbs, Adjectives).
4. Performs dimensionality reduction (t-SNE/PCA) to visualize global topological structures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
import argparse

import nltk
from nltk.corpus import brown

# We will use sklearn for dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from fourier_embeddings import FourierEmbedding

# Ensure NLTK resources are available
nltk.download('brown', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def load_brown_corpus(vocab_size=4000, sentence_limit=30000):
    print("Loading NLTK Brown corpus...")
    raw_sentences = brown.sents()
    if sentence_limit > 0:
        raw_sentences = raw_sentences[:sentence_limit]
        
    print(f"Loaded {len(raw_sentences)} sentences.")
    
    # Tokenize: keep only lowercase alphanumeric words
    processed_sentences = []
    word_counts = Counter()
    
    for sent in raw_sentences:
        tokens = []
        for w in sent:
            w_lower = w.lower()
            if w_lower.isalpha() and len(w_lower) > 1: # Only alpha, length > 1
                tokens.append(w_lower)
                word_counts[w_lower] += 1
        if tokens:
            processed_sentences.append(tokens)
            
    # Build vocabulary
    most_common = [w for w, c in word_counts.most_common(vocab_size)]
    vocab = sorted(list(most_common))
    
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for i, w in enumerate(vocab)}
    
    print(f"Vocabulary Size: {len(vocab)} words.")
    
    # Smooth unigram count for negative sampling
    counts = np.array([word_counts[w] + 1 for w in vocab], dtype=np.float32)
    smooth_counts = counts ** 0.75
    unigram_dist = smooth_counts / smooth_counts.sum()
    
    # Generate Skip-gram pairs
    pairs = []
    window_size = 3
    for sent in processed_sentences:
        sent_ids = [word2id[w] for w in sent if w in word2id]
        for i, target in enumerate(sent_ids):
            start = max(0, i - window_size)
            end = min(len(sent_ids), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    pairs.append((target, sent_ids[j]))
                    
    print(f"Generated {len(pairs)} skip-gram pairs.")
    return pairs, vocab, word2id, id2word, torch.tensor(unigram_dist, dtype=torch.float32)

# =============================================================================
# MODEL DEFINITION
# =============================================================================

class SpectralSGNS(nn.Module):
    def __init__(self, vocab_size, num_modes, init_scale=2.0):
        super().__init__()
        self.target_embed = FourierEmbedding(vocab_size, num_modes, init_scale)
        self.context_embed = FourierEmbedding(vocab_size, num_modes, init_scale)
        
    def forward(self, target_ids, context_ids, negative_ids):
        v_target = self.target_embed(target_ids)       # (B, D)
        v_context = self.context_embed(context_ids)     # (B, D)
        v_negatives = self.context_embed(negative_ids)  # (B, K, D)
        
        pos_scores = torch.sum(v_target * v_context, dim=-1)  # (B,)
        neg_scores = torch.bmm(v_negatives, v_target.unsqueeze(-1)).squeeze(-1) # (B, K)
        
        return pos_scores, neg_scores

# =============================================================================
# TOPOLOGICAL EVALUATION
# =============================================================================

def get_pos_categories(vocab):
    """Assign Part-of-Speech tags to the vocabulary using NLTK."""
    print("Assigning POS tags to vocabulary...")
    # NLTK pos_tag expects a list of words
    tagged = nltk.pos_tag(vocab)
    
    pos_map = {}
    for word, tag in tagged:
        if tag.startswith('NN'):
            pos_map[word] = 'Noun'
        elif tag.startswith('VB'):
            pos_map[word] = 'Verb'
        elif tag.startswith('JJ'):
            pos_map[word] = 'Adjective'
        else:
            pos_map[word] = 'Other'
            
    return pos_map

def plot_topology(model, vocab, id2word, pos_map, num_modes, device):
    """Perform t-SNE on the Fourier coefficients and plot by POS."""
    print("Extracting Fourier coefficients for topological analysis...")
    model.eval()
    
    # We will look at the raw Fourier coefficients (a_0, a_n, b_n)
    all_ids = torch.arange(len(vocab), device=device)
    with torch.no_grad():
        a0, an, bn = model.target_embed.get_raw_coefficients(all_ids)
        a0 = a0.cpu().numpy() # (V, 1)
        an = an.cpu().numpy() # (V, N)
        bn = bn.cpu().numpy() # (V, N)
        
    # Concatenate all coefficients into a single feature vector per word
    # Shape: (V, 2N + 1)
    coeffs = np.concatenate([a0, an, bn], axis=-1)
    
    print("Performing t-SNE dimensionality reduction...")
    # Use PCA first if dimensions are too high, but 2N+1 (e.g., 33) is fine for t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    coords = tsne.fit_transform(coeffs)
    
    print("Generating topological plot...")
    plt.figure(figsize=(12, 10), facecolor='#f8f9fa')
    
    # Colors for POS tags
    color_scheme = {
        'Noun': '#1a73e8',       # Blue
        'Verb': '#d93025',       # Red
        'Adjective': '#137333',  # Green
        'Other': '#bdc3c7'       # Gray (we will plot these smaller and more transparent)
    }
    
    # Plot 'Other' first so they stay in background
    for cat in ['Other', 'Noun', 'Verb', 'Adjective']:
        indices = [i for i, w in enumerate(vocab) if pos_map[w] == cat]
        if not indices: continue
        
        x = coords[indices, 0]
        y = coords[indices, 1]
        
        alpha = 0.2 if cat == 'Other' else 0.7
        size = 15 if cat == 'Other' else 35
        zorder = 1 if cat == 'Other' else 2
        
        plt.scatter(x, y, c=color_scheme[cat], label=cat, alpha=alpha, s=size, zorder=zorder, edgecolors='none')
        
    # Annotate a few representative words from each category
    np.random.seed(42)
    for cat in ['Noun', 'Verb', 'Adjective']:
        cat_indices = [i for i, w in enumerate(vocab) if pos_map[w] == cat]
        if not cat_indices: continue
        
        # Pick 10 random words to label
        sample_indices = np.random.choice(cat_indices, min(10, len(cat_indices)), replace=False)
        for idx in sample_indices:
            word = vocab[idx]
            plt.annotate(word, (coords[idx, 0], coords[idx, 1]), 
                         fontsize=9, alpha=0.9,
                         xytext=(3, 3), textcoords='offset points')
                         
    plt.title("Topological Emergence of Grammar in Fourier Space", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(frameon=True, facecolor='white', edgecolor='gray', fontsize=12, markerscale=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/topology_fourier_space.png", dpi=300, bbox_inches='tight')
    print("Saved topological plot to results/topology_fourier_space.png")
    plt.close()

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=int, default=16, help="Fourier modes")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--reg_weight", type=float, default=0.01, help="Spectral decay weight")
    parser.add_argument("--vocab_size", type=int, default=4000, help="Vocabulary size")
    parser.add_argument("--sentence_limit", type=int, default=30000, help="Sentences to read from Brown")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # 1. Load data
    pairs, vocab, word2id, id2word, unigram_dist = load_brown_corpus(args.vocab_size, args.sentence_limit)
    pos_map = get_pos_categories(vocab)
    
    targets_tensor = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    contexts_tensor = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    num_pairs = len(pairs)
    unigram_dist = unigram_dist.to(device)
    
    # 2. Define Model
    model = SpectralSGNS(len(vocab), args.modes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    harmonic_n = torch.arange(1, args.modes + 1, dtype=torch.float32, device=device)
    reg_weights = harmonic_n ** 1.8
    
    # 3. Training Loop
    print("\nTraining Fourier Embeddings directly from raw Brown corpus...")
    num_negatives = 5
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        
        indices = torch.randperm(num_pairs)
        
        for i in range(0, num_pairs, args.batch_size):
            batch_idx = indices[i:i + args.batch_size]
            B_size = len(batch_idx)
            
            xb = targets_tensor[batch_idx].to(device)
            yb = contexts_tensor[batch_idx].to(device)
            
            neg = torch.multinomial(unigram_dist, B_size * num_negatives, replacement=True).view(B_size, num_negatives).to(device)
            
            optimizer.zero_grad()
            pos_scores, neg_scores = model(xb, yb, neg)
            
            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()
            neg_loss = -torch.log(torch.sigmoid(-neg_scores) + 1e-8).sum(dim=-1).mean()
            loss = pos_loss + neg_loss
            
            # Spectral decay regularization
            _, an, bn = model.target_embed.get_raw_coefficients(xb)
            spectral_energy = (an**2 + bn**2) * reg_weights.unsqueeze(0)
            reg_loss = args.reg_weight * spectral_energy.mean()
            
            total_loss_with_reg = loss + reg_loss
            total_loss_with_reg.backward()
            optimizer.step()
            
            total_loss += total_loss_with_reg.item()
            total_batches += 1
            
        if epoch <= 3 or epoch % 10 == 0 or epoch == args.epochs:
            avg_loss = total_loss / total_batches
            print(f"  Epoch {epoch:02d}/{args.epochs:02d} | Loss: {avg_loss:.4f}")
            
    # 4. Topological Visualization
    plot_topology(model, vocab, id2word, pos_map, args.modes, device)

if __name__ == "__main__":
    main()
