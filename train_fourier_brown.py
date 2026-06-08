"""
train_fourier_brown.py — Learning Spectral Word Embeddings from Raw Text (Brown Corpus)
=====================================================================================
1. Downloads and loads the standard NLTK Brown corpus (~1.1M tokens).
2. Tokenizes and builds a vocabulary of the top words.
3. Generates Skip-gram pairs and runs fast GPU-accelerated (MPS) training in PyTorch.
4. Evaluates analogies, singular-to-plural phase shifts, and circular convolution composition.
5. Saves clean wave visualizations in the results/ directory.
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

from fourier_embeddings import FourierEmbedding, PhaseShiftOperator, compose_circular_convolution

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def load_brown_corpus(vocab_size=3000, sentence_limit=25000):
    print("Loading NLTK Brown corpus...")
    nltk.download('brown', quiet=True)
    raw_sentences = brown.sents()
    if sentence_limit > 0:
        raw_sentences = raw_sentences[:sentence_limit]
        
    print(f"Loaded {len(raw_sentences)} sentences.")
    
    # Tokenize: keep only lowercase alphanumeric words
    processed_sentences = []
    word_counts = Counter()
    
    # Target evaluation words to force into vocabulary
    forced_words = {
        # Analogy words
        "king", "queen", "man", "woman", "prince", "princess", "brother", "sister", 
        "father", "mother", "uncle", "aunt", "son", "daughter", "husband", "wife",
        "paris", "france", "london", "uk", "rome", "italy", "tokyo", "japan", 
        "berlin", "germany", "madrid", "spain", "beijing", "china",
        # Grammar words
        "dog", "dogs", "cat", "cats", "car", "cars", "tree", "trees", "house", "houses", 
        "bird", "birds", "book", "books", "friend", "friends", "boy", "boys",
        "walk", "walked", "run", "ran", "go", "went", "play", "played", 
        # Composition words
        "red", "green", "blue", "apple", "car"
    }
    
    for sent in raw_sentences:
        tokens = []
        for w in sent:
            w_lower = w.lower()
            if w_lower.isalnum():
                tokens.append(w_lower)
                word_counts[w_lower] += 1
        if tokens:
            processed_sentences.append(tokens)
            
    # Build vocabulary: top words + forced words
    most_common = [w for w, c in word_counts.most_common(vocab_size)]
    vocab_set = set(most_common) | forced_words
    vocab = sorted(list(vocab_set))
    
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for i, w in enumerate(vocab)}
    
    print(f"Vocabulary Size: {len(vocab)} words (including {len(forced_words)} forced evaluation words).")
    
    # Smooth unigram count for negative sampling (3/4 power law)
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
# EVALUATION METRICS
# =============================================================================

def find_nearest_neighbor(query_vec, all_embeddings, exclude_ids=None):
    dot_products = torch.matmul(all_embeddings, query_vec)
    norms = torch.norm(all_embeddings, dim=-1) * torch.norm(query_vec)
    cos_sims = dot_products / (norms + 1e-8)
    
    if exclude_ids is not None:
        for idx in exclude_ids:
            cos_sims[idx] = -2.0
            
    return cos_sims.argmax().item(), cos_sims

def evaluate_analogies(model, word2id, id2word, device):
    analogies = [
        # Gender
        ("king", "man", "woman", "queen"),
        ("queen", "woman", "man", "king"),
        ("prince", "boy", "girl", "princess"),
        ("princess", "girl", "boy", "prince"),
        ("brother", "man", "woman", "sister"),
        ("sister", "woman", "man", "brother"),
        ("father", "man", "woman", "mother"),
        ("mother", "woman", "man", "father"),
        # Capitals
        ("paris", "france", "uk", "london"),
        ("london", "uk", "france", "paris"),
        ("rome", "italy", "japan", "tokyo"),
        ("tokyo", "japan", "italy", "rome"),
        ("berlin", "germany", "france", "paris"),
    ]
    
    model.eval()
    all_embeds = model.target_embed(torch.arange(len(word2id), device=device))
    
    correct = 0
    total = 0
    
    print("\nEvaluating Analogies:")
    for wA, wB, wC, wD in analogies:
        if all(w in word2id for w in [wA, wB, wC, wD]):
            idA, idB, idC, idD = word2id[wA], word2id[wB], word2id[wC], word2id[wD]
            vA, vB, vC = all_embeds[idA], all_embeds[idB], all_embeds[idC]
            
            # Target = A - B + C
            v_target = vA - vB + vC
            pred_id, sims = find_nearest_neighbor(v_target, all_embeds, exclude_ids=[idA, idB, idC])
            
            is_correct = (pred_id == idD)
            if is_correct:
                correct += 1
            total += 1
            print(f"  {wA:8s} - {wB:6s} + {wC:6s} = {id2word[pred_id]:8s} (Target: {wD:8s}) [{'CORRECT' if is_correct else 'WRONG'}]")
            
    acc = correct / total if total > 0 else 0.0
    print(f"--> Analogy Accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# =============================================================================
# ROTATIONS EVALUATION
# =============================================================================

def evaluate_rotations(model, word2id, id2word, word_pairs, name, num_modes, device):
    model.eval()
    valid_pairs = [(word2id[w1], word2id[w2]) for w1, w2 in word_pairs if w1 in word2id and w2 in word2id]
    if not valid_pairs:
        return 0.0
        
    src_ids = torch.tensor([p[0] for p in valid_pairs], dtype=torch.long, device=device)
    tgt_ids = torch.tensor([p[1] for p in valid_pairs], dtype=torch.long, device=device)
    
    with torch.no_grad():
        v_src = model.target_embed(src_ids)
        v_tgt = model.target_embed(tgt_ids)
        
    rotator = PhaseShiftOperator(num_modes, mode='independent').to(device)
    optimizer = optim.Adam(rotator.parameters(), lr=0.1)
    
    for epoch in range(120):
        optimizer.zero_grad()
        v_pred = rotator(v_src)
        loss = nn.MSELoss()(v_pred, v_tgt)
        loss.backward()
        optimizer.step()
        
    correct = 0
    total = len(valid_pairs)
    all_embeds = model.target_embed(torch.arange(len(word2id), device=device))
    
    print(f"\nEvaluating Grammatical Rotations ({name}):")
    with torch.no_grad():
        v_shifted = rotator(v_src)
        for i in range(total):
            src_id = src_ids[i].item()
            tgt_id = tgt_ids[i].item()
            pred_id, sims = find_nearest_neighbor(v_shifted[i], all_embeds, exclude_ids=[src_id])
            is_correct = (pred_id == tgt_id)
            if is_correct:
                correct += 1
            print(f"  Rotated({id2word[src_id]:8s}) -> {id2word[pred_id]:8s} (Target: {id2word[tgt_id]:8s}) [{'CORRECT' if is_correct else 'WRONG'}]")
            
    acc = correct / total
    print(f"--> Rotation Retrieval Accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=int, default=16, help="Fourier modes")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--reg_weight", type=float, default=0.03, help="Spectral decay weight")
    parser.add_argument("--reg_power", type=float, default=1.8, help="Spectral penalty power")
    parser.add_argument("--vocab_size", type=int, default=4000, help="Vocabulary size")
    parser.add_argument("--sentence_limit", type=int, default=30000, help="Sentences to read from Brown")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # GPU / MPS Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # 1. Load data
    pairs, vocab, word2id, id2word, unigram_dist = load_brown_corpus(args.vocab_size, args.sentence_limit)
    
    # Load all pairs into tensors for fast batching
    targets_tensor = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    contexts_tensor = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    num_pairs = len(pairs)
    
    # Move unigram dist to device for vectorized sampling
    unigram_dist = unigram_dist.to(device)
    
    # 2. Define Model
    model = SpectralSGNS(len(vocab), args.modes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Regularization frequency weights
    harmonic_n = torch.arange(1, args.modes + 1, dtype=torch.float32, device=device)
    reg_weights = harmonic_n ** args.reg_power
    
    # 3. Training Loop
    print("\nTraining Fourier Embeddings directly from raw Brown corpus...")
    num_negatives = 5
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        
        # Shuffle indices
        indices = torch.randperm(num_pairs)
        
        for i in range(0, num_pairs, args.batch_size):
            batch_idx = indices[i:i + args.batch_size]
            B_size = len(batch_idx)
            
            xb = targets_tensor[batch_idx].to(device)
            yb = contexts_tensor[batch_idx].to(device)
            
            # GPU-accelerated negative sampling via multinomial draw
            neg = torch.multinomial(unigram_dist, B_size * num_negatives, replacement=True).view(B_size, num_negatives).to(device)
            
            optimizer.zero_grad()
            pos_scores, neg_scores = model(xb, yb, neg)
            
            # Loss computation
            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()
            neg_loss = -torch.log(torch.sigmoid(-neg_scores) + 1e-8).sum(dim=-1).mean()
            loss = pos_loss + neg_loss
            
            # Spectral decay regularization to force smooth wave functions
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
            print(f"  Epoch {epoch:02d}/{args.epochs:02d} | Loss: {avg_loss:.4f} | Scale P: {model.target_embed.scale.item():.4f}")
            
    # 4. Evaluate Analogies
    evaluate_analogies(model, word2id, id2word, device)
    
    # 5. Evaluate Rotations (Grammar)
    plurals = [
        ("dog", "dogs"), ("cat", "cats"), ("car", "cars"), ("tree", "trees"),
        ("house", "houses"), ("bird", "birds"), ("book", "books")
    ]
    evaluate_rotations(model, word2id, id2word, plurals, "Singular-to-Plural", args.modes, device)
    
    tenses = [
        ("walk", "walked"), ("run", "ran"), ("go", "went"), ("play", "played")
    ]
    evaluate_rotations(model, word2id, id2word, tenses, "Present-to-Past Tense", args.modes, device)
    
    # 6. Compositionality Evaluation
    model.eval()
    if "red" in word2id and "apple" in word2id and "green" in word2id and "car" in word2id and "blue" in word2id:
        v_red = model.target_embed(torch.tensor(word2id["red"], device=device))
        v_green = model.target_embed(torch.tensor(word2id["green"], device=device))
        v_blue = model.target_embed(torch.tensor(word2id["blue"], device=device))
        v_apple = model.target_embed(torch.tensor(word2id["apple"], device=device))
        v_car = model.target_embed(torch.tensor(word2id["car"], device=device))
        
        # Convolve
        red_apple = compose_circular_convolution(v_red, v_apple, args.modes)
        green_apple = compose_circular_convolution(v_green, v_apple, args.modes)
        blue_car = compose_circular_convolution(v_blue, v_car, args.modes)
        
        def cos_sim(vA, vB):
            return torch.dot(vA, vB) / (torch.norm(vA) * torch.norm(vB) + 1e-8)
            
        print("\nEvaluating Semantic Compositionality on Brown Fourier Embeddings:")
        sim1 = cos_sim(red_apple, green_apple).item()
        sim2 = cos_sim(red_apple, blue_car).item()
        print(f"  Sim(red_apple, green_apple) = {sim1:.4f} (Same noun category)")
        print(f"  Sim(red_apple, blue_car)    = {sim2:.4f} (Different category)")
        
    # 7. Visualization: Wave profiles
    print("\nGenerating final wave profiles for visualization in results/ directory...")
    t = torch.linspace(0.0, float(model.target_embed.scale.abs().item()), steps=500, device=device)
    
    words_to_plot = ["king", "queen", "man", "woman", "dog", "dogs"]
    plotted_words = [w for w in words_to_plot if w in word2id]
    
    if plotted_words:
        word_ids = torch.tensor([word2id[w] for w in plotted_words], dtype=torch.long, device=device)
        with torch.no_grad():
            f_waves = model.target_embed.reconstruct_function(word_ids, t).cpu().numpy()
            
        t_np = t.cpu().numpy()
        
        plt.figure(figsize=(10, 7), facecolor='#f8f9fa')
        
        # Plot 1: Semantic waves
        plt.subplot(2, 1, 1)
        colors = {
            "king": "#1a73e8",   # blue
            "queen": "#d93025",  # red
            "man": "#8ab4f8",    # light blue
            "woman": "#f28b82"   # light red
        }
        for i, word in enumerate(plotted_words):
            if word in colors:
                style = '-' if word in ["king", "queen"] else '--'
                plt.plot(t_np, f_waves[i], label=word, color=colors[word], linestyle=style, linewidth=2.5)
                
        plt.title("Semantic Wave Profiles Learnt from Raw Text (Brown Corpus)", fontsize=12, fontweight='bold', pad=10)
        plt.xlabel("Domain Coordinate t", fontsize=10)
        plt.ylabel("Function Value f(t)", fontsize=10)
        plt.legend(frameon=True, facecolor='white', edgecolor='none')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Plot 2: Grammatical Shift
        plt.subplot(2, 1, 2)
        colors_grammar = {
            "dog": "#137333",    # green
            "dogs": "#e37400"    # orange
        }
        for i, word in enumerate(plotted_words):
            if word in colors_grammar:
                plt.plot(t_np, f_waves[i], label=word, color=colors_grammar[word], linewidth=2.5)
                
        plt.title("Grammatical Shift (Singular vs. Plural)", fontsize=12, fontweight='bold', pad=10)
        plt.xlabel("Domain Coordinate t", fontsize=10)
        plt.ylabel("Function Value f(t)", fontsize=10)
        plt.legend(frameon=True, facecolor='white', edgecolor='none')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/brown_fourier_waves.png", dpi=150, bbox_inches='tight')
        print("Saved wave profiles plot to results/brown_fourier_waves.png")
        plt.close()

if __name__ == "__main__":
    main()
