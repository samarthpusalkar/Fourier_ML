"""
compress_glove_fourier.py — Fourier Compression of Stanford GloVe Embeddings
========================================================================
1. Downloads the 50-dimensional GloVe embeddings from Hugging Face.
2. Projects the 50d vectors into 33d Fourier coefficients (16 modes).
3. Compares analogy and grammar rotation performance before and after compression.
4. Generates a beautiful and clean word wave visualization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datasets import load_dataset
import sys

from fourier_embeddings import PhaseShiftOperator, compose_circular_convolution

# =============================================================================
# 1. LOAD AND PARSE GLOVE EMBEDDINGS (STREAMING FROM HF)
# =============================================================================

def load_glove_subset(num_words=10000):
    print(f"Streaming top {num_words} GloVe vectors from Hugging Face...")
    ds = load_dataset("Jay-Mayekar/glove-vectors", split="train", streaming=True)
    
    word2vec = {}
    word2id = {}
    id2word = {}
    
    count = 0
    for row in ds:
        parts = row['text'].strip().split()
        if len(parts) < 2:
            continue
        word = parts[0]
        # Keep word lowercase for consistency
        word = word.lower()
        
        # Parse vector
        vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        
        # Store if not already stored
        if word not in word2vec:
            word2vec[word] = vec
            word2id[word] = count
            id2word[count] = word
            count += 1
            
        if count >= num_words:
            break
            
    print(f"Loaded {len(word2vec)} words with dimension {next(iter(word2vec.values())).shape[0]}.")
    return word2vec, word2id, id2word

# =============================================================================
# 2. FOURIER PROJECTION MATRICES
# =============================================================================

class FourierCompressor:
    """
    Compresses discrete D-dimensional vectors to 2N+1 Fourier coefficients.
    Reconstructs continuous periodic functions f(t).
    """
    def __init__(self, input_dim, num_modes):
        self.D = input_dim
        self.N = num_modes
        
        # Setup discrete grid points t_j in [0, 1)
        t = np.linspace(0.0, 1.0, self.D, endpoint=False)
        
        # Construct the basis matrix A of shape (D, 1 + 2*N)
        A = []
        A.append(np.ones_like(t)) # a0
        for n in range(1, num_modes + 1):
            A.append(np.cos(2.0 * np.pi * n * t) / np.sqrt(2.0)) # a_n scaled
            A.append(np.sin(2.0 * np.pi * n * t) / np.sqrt(2.0)) # b_n scaled
            
        self.A = np.stack(A, axis=1) # (D, 2N+1)
        
        # Pseudo-inverse for projection: C = pinv(A) * X
        self.A_pinv = np.linalg.pinv(self.A) # (2N+1, D)

    def compress(self, vec):
        """
        Projects a D-dimensional vector to 2N+1 Fourier coefficients.
        Returns scaled coefficients (Euclidean dot product matches L2 continuous).
        """
        # C is of shape (2N+1,) representing:
        # [a0, a_1 / sqrt(2), b_1 / sqrt(2), ..., a_N / sqrt(2), b_N / sqrt(2)]
        return np.dot(self.A_pinv, vec)

    def reconstruct_continuous(self, coeffs, num_points=500):
        """
        Evaluates the continuous wave f(t) reconstructed from the coefficients.
        """
        t = np.linspace(0.0, 1.0, num_points)
        a0 = coeffs[0]
        
        # Deconstruct coefficients (need raw unscaled coefficients for formulas)
        # Note: In our matrix A, the columns for cosines/sines were scaled by 1/sqrt(2).
        # Therefore, the coefficients returned by least squares already account for it.
        # We can evaluate directly using the columns:
        f_t = np.zeros_like(t) + a0
        for n in range(1, self.N + 1):
            an_scaled = coeffs[2*n - 1]
            bn_scaled = coeffs[2*n]
            # Since basis was scaled by 1/sqrt(2), to evaluate:
            f_t += (an_scaled / np.sqrt(2.0)) * np.cos(2.0 * np.pi * n * t) * 2.0
            f_t += (bn_scaled / np.sqrt(2.0)) * np.sin(2.0 * np.pi * n * t) * 2.0
            
        return t, f_t

# =============================================================================
# 3. EVALUATION FUNCTION
# =============================================================================

def find_nearest_neighbor(query, matrix, exclude_ids=None):
    dot_products = np.dot(matrix, query)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    sims = dot_products / (norms + 1e-8)
    
    if exclude_ids is not None:
        for idx in exclude_ids:
            sims[idx] = -2.0
            
    return np.argmax(sims), sims

def evaluate_analogies(embeddings, word2id, id2word, name):
    analogies = [
        # Gender
        ("king", "man", "woman", "queen"),
        ("queen", "woman", "man", "king"),
        ("prince", "boy", "girl", "princess"),
        ("princess", "girl", "boy", "prince"),
        ("brother", "man", "woman", "sister"),
        ("sister", "woman", "man", "brother"),
        ("uncle", "man", "woman", "aunt"),
        ("aunt", "woman", "man", "uncle"),
        ("father", "man", "woman", "mother"),
        ("mother", "woman", "man", "father"),
        # Capitals
        ("paris", "france", "uk", "london"),
        ("london", "uk", "france", "paris"),
        ("rome", "italy", "japan", "tokyo"),
        ("tokyo", "japan", "italy", "rome"),
        ("berlin", "germany", "france", "paris"),
        ("madrid", "spain", "france", "paris"),
        ("beijing", "china", "japan", "tokyo"),
    ]
    
    valid_analogies = []
    for wA, wB, wC, wD in analogies:
        if all(w in word2id for w in [wA, wB, wC, wD]):
            valid_analogies.append((wA, wB, wC, wD))
            
    if not valid_analogies:
        print(f"No valid analogies found in the subset for {name}!")
        return 0.0
        
    correct = 0
    total = len(valid_analogies)
    
    print(f"\nEvaluating Analogies on {name} (Total evaluated: {total}):")
    for wA, wB, wC, wD in valid_analogies:
        idA, idB, idC, idD = word2id[wA], word2id[wB], word2id[wC], word2id[wD]
        vA, vB, vC = embeddings[idA], embeddings[idB], embeddings[idC]
        
        # Math: v_target = vA - vB + vC
        v_target = vA - vB + vC
        
        pred_id, sims = find_nearest_neighbor(v_target, embeddings, exclude_ids=[idA, idB, idC])
        
        is_correct = (pred_id == idD)
        if is_correct:
            correct += 1
            
        print(f"  {wA:8s} - {wB:6s} + {wC:6s} = {id2word[pred_id]:8s} (Target: {wD:8s}) [{'CORRECT' if is_correct else 'WRONG'}]")
        
    acc = correct / total
    print(f"--> Analogy Accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# =============================================================================
# 4. GRAMMAR ROTATION TRAINING
# =============================================================================

def evaluate_grammar_rotations(embeddings, word2id, id2word, word_pairs, num_modes, name):
    valid_pairs = [(word2id[w1], word2id[w2]) for w1, w2 in word_pairs if w1 in word2id and w2 in word2id]
    if not valid_pairs:
        print(f"No valid pairs for rotation evaluation of {name}.")
        return 0.0
        
    src_ids = torch.tensor([p[0] for p in valid_pairs], dtype=torch.long)
    tgt_ids = torch.tensor([p[1] for p in valid_pairs], dtype=torch.long)
    
    embed_tensor = torch.tensor(embeddings, dtype=torch.float32)
    v_src = embed_tensor[src_ids]
    v_tgt = embed_tensor[tgt_ids]
    
    rotator = PhaseShiftOperator(num_modes, mode='independent')
    optimizer = optim.Adam(rotator.parameters(), lr=0.1)
    
    # Train rotator to rotate source to target
    for epoch in range(100):
        optimizer.zero_grad()
        v_pred = rotator(v_src)
        loss = nn.MSELoss()(v_pred, v_tgt)
        loss.backward()
        optimizer.step()
        
    correct = 0
    total = len(valid_pairs)
    
    print(f"\nEvaluating Grammatical Rotations on {name}:")
    with torch.no_grad():
        v_shifted = rotator(v_src).numpy()
        for i in range(total):
            src_id = src_ids[i].item()
            tgt_id = tgt_ids[i].item()
            
            pred_id, sims = find_nearest_neighbor(v_shifted[i], embeddings, exclude_ids=[src_id])
            is_correct = (pred_id == tgt_id)
            if is_correct:
                correct += 1
            print(f"  Rotated({id2word[src_id]:8s}) -> {id2word[pred_id]:8s} (Target: {id2word[tgt_id]:8s}) [{'CORRECT' if is_correct else 'WRONG'}]")
            
    acc = correct / total
    print(f"--> Rotation Retrieval Accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# =============================================================================
# MAIN EXPERIMENT RUN
# =============================================================================

def main():
    # Load GloVe
    word2vec, word2id, id2word = load_glove_subset(num_words=10000)
    
    # Embeddings matrix
    vocab_size = len(word2id)
    D = next(iter(word2vec.values())).shape[0]
    glove_matrix = np.stack([word2vec[id2word[i]] for i in range(vocab_size)])
    
    # 2. Fourier Compressor Setup
    num_modes = 16 # 16 modes = 33 dimensions
    compressor = FourierCompressor(D, num_modes)
    
    # Compress all GloVe vectors to Fourier coefficient space
    print(f"\nCompressing 50-dimensional GloVe to 33-dimensional Fourier coefficients (Compression Ratio: {33/50:.2f})...")
    fourier_matrix = np.stack([compressor.compress(glove_matrix[i]) for i in range(vocab_size)])
    
    # 3. Evaluate Analogies
    # Original GloVe
    evaluate_analogies(glove_matrix, word2id, id2word, "Original 50d GloVe")
    # Compressed Fourier
    evaluate_analogies(fourier_matrix, word2id, id2word, "Compressed 33d Fourier waves")
    
    # 4. Evaluate Grammatical Rotations (Plurals)
    plurals = [
        ("dog", "dogs"), ("cat", "cats"), ("car", "cars"), ("tree", "trees"),
        ("house", "houses"), ("bird", "birds"), ("apple", "apples"),
        ("book", "books"), ("friend", "friends"), ("boy", "boys"),
        ("child", "children"), ("man", "men"), ("woman", "women"),
        ("year", "years"), ("day", "days"), ("hand", "hands"),
        ("place", "places"), ("part", "parts"), ("life", "lives")
    ]
    evaluate_grammar_rotations(fourier_matrix, word2id, id2word, plurals, num_modes, "Singular-to-Plural (Rotations)")
    
    # 5. Evaluate Compositionality via circular convolution
    # Pointwise complex multiplication of Fourier coefficients
    phrases = {
        "red_apple": ("red", "apple"),
        "green_apple": ("green", "apple"),
        "blue_car": ("blue", "car"),
    }
    
    embeds_composed = {}
    for key, (w1, w2) in phrases.items():
        if w1 in word2id and w2 in word2id:
            v1 = torch.tensor(fourier_matrix[word2id[w1]], dtype=torch.float32)
            v2 = torch.tensor(fourier_matrix[word2id[w2]], dtype=torch.float32)
            # pointwise complex multiplication
            v_conv = compose_circular_convolution(v1, v2, num_modes).numpy()
            embeds_composed[key] = v_conv
            
    if len(embeds_composed) >= 3:
        def cos_sim(vA, vB):
            return np.dot(vA, vB) / (np.linalg.norm(vA) * np.linalg.norm(vB) + 1e-8)
            
        print("\nEvaluating Semantic Compositionality on Compressed Fourier Embeddings:")
        sim1 = cos_sim(embeds_composed["red_apple"], embeds_composed["green_apple"])
        sim2 = cos_sim(embeds_composed["red_apple"], embeds_composed["blue_car"])
        print(f"  Sim(red_apple, green_apple) = {sim1:.4f} (Same noun category)")
        print(f"  Sim(red_apple, blue_car)    = {sim2:.4f} (Completely different category)")
        
    # 6. Generate BEAUTIFUL wave profile visualization
    print("\nReconstructing smooth continuous wave profiles f(t) for visualization...")
    words_to_plot = ["king", "queen", "man", "woman", "dog", "dogs"]
    
    plt.figure(figsize=(10, 7), facecolor='#f8f9fa')
    
    # Plot 1: Semantic waves
    plt.subplot(2, 1, 1)
    colors = {
        "king": "#1a73e8",   # Deep blue
        "queen": "#d93025",  # Deep red
        "man": "#8ab4f8",    # Light blue (dashed)
        "woman": "#f28b82"   # Light red (dashed)
    }
    for w in ["king", "queen", "man", "woman"]:
        if w in word2id:
            coeffs = fourier_matrix[word2id[w]]
            t, f_t = compressor.reconstruct_continuous(coeffs)
            style = '-' if w in ["king", "queen"] else '--'
            plt.plot(t, f_t, label=w, color=colors[w], linestyle=style, linewidth=2.5)
            
    plt.title("Semantic Wave Profiles from Compressed GloVe", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Domain Coordinate t", fontsize=10)
    plt.ylabel("Function Value f(t)", fontsize=10)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: Grammatical Shift
    plt.subplot(2, 1, 2)
    colors_grammar = {
        "dog": "#137333",    # Green
        "dogs": "#e37400"    # Orange
    }
    for w in ["dog", "dogs"]:
        if w in word2id:
            coeffs = fourier_matrix[word2id[w]]
            t, f_t = compressor.reconstruct_continuous(coeffs)
            plt.plot(t, f_t, label=w, color=colors_grammar[w], linewidth=2.5)
            
    plt.title("Grammatical Shift Wave Profiles (Singular vs. Plural)", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Domain Coordinate t", fontsize=10)
    plt.ylabel("Function Value f(t)", fontsize=10)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("glove_fourier_waves.png", dpi=150, bbox_inches='tight')
    print("Saved beautiful wave profiles plot to glove_fourier_waves.png")
    plt.close()

if __name__ == "__main__":
    main()
