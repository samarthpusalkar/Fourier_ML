"""
train_fourier_embeddings.py — Main Experiment Script for Spectral Word Embeddings
================================================================================
Generates a structured grammatical synthetic dataset, trains Fourier embeddings,
evaluates semantic/grammatical algebra, and visualizes learned representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from collections import Counter

from fourier_embeddings import FourierEmbedding, PhaseShiftOperator, compose_circular_convolution

# =============================================================================
# DATASET GENERATION
# =============================================================================

def get_grammatical_dataset():
    # 1. Word mappings defining relations
    plurals = [
        ("dog", "dogs"), ("cat", "cats"), ("car", "cars"), ("tree", "trees"),
        ("house", "houses"), ("bird", "birds"), ("apple", "apples"),
        ("book", "books"), ("friend", "friends"), ("boy", "boys")
    ]
    
    gender = [
        ("man", "woman"), ("king", "queen"), ("boy", "girl"), ("actor", "actress"),
        ("brother", "sister"), ("father", "mother"), ("uncle", "aunt"),
        ("son", "daughter"), ("prince", "princess"), ("husband", "wife")
    ]
    
    tenses = [
        ("walk", "walked"), ("run", "ran"), ("go", "went"), ("play", "played"),
        ("talk", "talked"), ("jump", "jumped"), ("sing", "sang"), ("read", "read"),
        ("write", "wrote"), ("see", "saw")
    ]
    
    capitals = [
        ("paris", "france"), ("london", "uk"), ("rome", "italy"), ("tokyo", "japan"),
        ("berlin", "germany"), ("madrid", "spain"), ("beijing", "china"),
        ("ottawa", "canada"), ("cairo", "egypt"), ("delhi", "india")
    ]
    
    # Adjectives and nouns for composition
    adjectives = ["red", "blue", "green", "large", "small", "hot", "cold", "fast", "slow", "sweet"]
    nouns = ["apple", "car", "water", "house", "wind", "fruit", "vehicle", "object"]
    
    # Generate rich co-occurrence sentences
    templates = []
    
    # Semantic relation sentences
    for s, p in plurals:
        templates.append(f"the {s} runs and the {p} run")
        templates.append(f"a {s} lives here and many {p} live there")
    
    for m, w in gender:
        templates.append(f"the {m} is family and the {w} is family")
        templates.append(f"a {m} acts and a {w} acts too")
    templates.append("the man is a king and the woman is a queen")
    templates.append("the boy is a prince and the girl is a princess")
    templates.append("brother and sister father and mother uncle and aunt son and daughter")
    templates.append("husband and wife actor and actress king and queen man and woman")
    
    for pres, past in tenses:
        templates.append(f"today I {pres} and yesterday I {past}")
        templates.append(f"he likes to {pres} and she {past} yesterday")
        
    for cap, country in capitals:
        templates.append(f"{cap} is the capital of {country}")
        templates.append(f"I visited {cap} in {country}")
        
    # Composition sentences
    templates.append("the red apple is a sweet fruit")
    templates.append("the blue car is a fast vehicle")
    templates.append("the green tree has large leaves")
    templates.append("hot water is a hot liquid")
    templates.append("cold wind is a cold breeze")
    templates.append("large house is a large building")
    templates.append("small bird is a small animal")
    templates.append("fast runner is a fast person")
    templates.append("slow progress is a slow speed")
    templates.append("sweet apple is a sweet fruit")
    templates.append("the green apple is a fresh fruit")
    templates.append("the red car is a fast vehicle")
    templates.append("the blue apple is a strange fruit")
    templates.append("the red water is hot")
    templates.append("the cold water is fresh")
    templates.append("the large car is slow")
    templates.append("the small car is fast")
    templates.append("the hot house is large")
    templates.append("the cold house is small")
    
    # Repeat sentences to increase corpus size and stabilize training
    sentences = []
    for _ in range(80):  # Repeat 80 times
        for s in templates:
            sentences.append(s.split())
            
    return sentences, plurals, gender, tenses, capitals

# =============================================================================
# TOKENIZATION AND SKIP-GRAM BUILDER
# =============================================================================

class TextDataset:
    def __init__(self, sentences, window_size=3, num_negatives=5):
        self.sentences = sentences
        self.window_size = window_size
        self.num_negatives = num_negatives
        
        # Build vocab
        all_words = [w for s in sentences for w in s]
        word_counts = Counter(all_words)
        self.vocab = sorted(list(word_counts.keys()))
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.id2word = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        # Unigram distribution for negative sampling
        counts = np.array([word_counts[w] for w in self.vocab], dtype=np.float32)
        # Smooth using 3/4 power law like standard Word2Vec
        smooth_counts = counts ** 0.75
        self.unigram_dist = smooth_counts / smooth_counts.sum()
        
        # Pre-generate a large pool of negative samples to speed up training
        self.neg_pool = np.random.choice(
            self.vocab_size,
            size=300000,
            p=self.unigram_dist
        )
        self.neg_ptr = 0
        
        # Generate skip-gram pairs
        self.pairs = []
        for sentence in sentences:
            sentence_ids = [self.word2id[w] for w in sentence]
            for i, target in enumerate(sentence_ids):
                start = max(0, i - window_size)
                end = min(len(sentence_ids), i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        self.pairs.append((target, sentence_ids[j]))
                        
    def get_batches(self, batch_size):
        np.random.shuffle(self.pairs)
        num_batches = len(self.pairs) // batch_size
        for i in range(num_batches):
            batch_pairs = self.pairs[i * batch_size:(i + 1) * batch_size]
            targets = torch.tensor([p[0] for p in batch_pairs], dtype=torch.long)
            contexts = torch.tensor([p[1] for p in batch_pairs], dtype=torch.long)
            
            # Fast negative sampling from pre-generated pool
            req_negatives = len(batch_pairs) * self.num_negatives
            if self.neg_ptr + req_negatives > len(self.neg_pool):
                # Regenerate pool if we reach the end
                self.neg_pool = np.random.choice(
                    self.vocab_size,
                    size=300000,
                    p=self.unigram_dist
                )
                self.neg_ptr = 0
            
            neg_samples = self.neg_pool[self.neg_ptr : self.neg_ptr + req_negatives].reshape(
                len(batch_pairs), self.num_negatives
            )
            self.neg_ptr += req_negatives
            negatives = torch.tensor(neg_samples, dtype=torch.long)
            
            yield targets, contexts, negatives

# =============================================================================
# MODEL DEFINITION
# =============================================================================

class SpectralSGNS(nn.Module):
    def __init__(self, vocab_size, num_modes, init_scale=2.0):
        super().__init__()
        self.target_embed = FourierEmbedding(vocab_size, num_modes, init_scale)
        self.context_embed = FourierEmbedding(vocab_size, num_modes, init_scale)
        
    def forward(self, target_ids, context_ids, negative_ids):
        # target_ids: (B,)
        # context_ids: (B,)
        # negative_ids: (B, K)
        
        v_target = self.target_embed(target_ids)       # (B, D)
        v_context = self.context_embed(context_ids)     # (B, D)
        v_negatives = self.context_embed(negative_ids)  # (B, K, D)
        
        # Positives scores
        pos_scores = torch.sum(v_target * v_context, dim=-1)  # (B,)
        
        # Negatives scores
        neg_scores = torch.bmm(v_negatives, v_target.unsqueeze(-1)).squeeze(-1) # (B, K)
        
        return pos_scores, neg_scores

# =============================================================================
# EVALUATION METRICS
# =============================================================================

def find_nearest_neighbor(query_vec, all_embeddings, exclude_ids=None):
    """
    Finds the index of the word with the highest cosine similarity to query_vec.
    """
    # query_vec: (D,)
    # all_embeddings: (V, D)
    dot_products = torch.matmul(all_embeddings, query_vec)
    norms = torch.norm(all_embeddings, dim=-1) * torch.norm(query_vec)
    cos_sims = dot_products / (norms + 1e-8)
    
    if exclude_ids is not None:
        for idx in exclude_ids:
            cos_sims[idx] = -2.0  # Set similarity very low to exclude
            
    return cos_sims.argmax().item(), cos_sims

def evaluate_analogies(model, word2id, id2word, analogies, relation_name):
    """
    Evaluates word analogies: wA - wB + wC = wD
    e.g. king - man + woman = queen
    """
    model.eval()
    all_embeds = model.target_embed(torch.arange(len(word2id), device=next(model.parameters()).device))
    
    correct = 0
    total = 0
    
    print(f"\nEvaluating Analogies ({relation_name}):")
    for pair in analogies:
        # We test both directions:
        # A - B + C = D (e.g. king - man + woman = queen)
        # D - C + B = A (e.g. queen - woman + man = king)
        
        for wA, wB, wC, wD in [
            (pair[0][0], pair[0][1], pair[1][1], pair[1][0]), # king - man + woman = queen
            (pair[1][0], pair[1][1], pair[0][1], pair[0][0])  # queen - woman + man = king
        ]:
            if all(w in word2id for w in [wA, wB, wC, wD]):
                idA, idB, idC, idD = word2id[wA], word2id[wB], word2id[wC], word2id[wD]
                
                vA = all_embeds[idA]
                vB = all_embeds[idB]
                vC = all_embeds[idC]
                
                v_target = vA - vB + vC
                
                pred_id, sims = find_nearest_neighbor(v_target, all_embeds, exclude_ids=[idA, idB, idC])
                
                if pred_id == idD:
                    correct += 1
                total += 1
                
                # Print sample
                if total <= 4:
                    print(f"  {wA} - {wB} + {wC} = {id2word[pred_id]} (Target: {wD}) [Sim: {sims[pred_id]:.4f}]")
                    
    acc = correct / total if total > 0 else 0.0
    print(f"  Analogy Accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# =============================================================================
# GRAMMAR ROTATION TRAINING
# =============================================================================

def train_grammar_rotation(model, word2id, id2word, word_pairs, name, epochs=30, mode='independent'):
    """
    Trains a PhaseShiftOperator to map the singular/present words to plural/past words.
    """
    model.eval()
    device = next(model.parameters()).device
    num_modes = model.target_embed.num_modes
    
    # Gather training pairs
    valid_pairs = [(word2id[w1], word2id[w2]) for w1, w2 in word_pairs if w1 in word2id and w2 in word2id]
    if not valid_pairs:
        return 0.0
        
    src_ids = torch.tensor([p[0] for p in valid_pairs], dtype=torch.long, device=device)
    tgt_ids = torch.tensor([p[1] for p in valid_pairs], dtype=torch.long, device=device)
    
    # Get frozen embeddings
    with torch.no_grad():
        v_src = model.target_embed(src_ids)  # (P, D)
        v_tgt = model.target_embed(tgt_ids)  # (P, D)
        
    # Define and train PhaseShiftOperator
    rotator = PhaseShiftOperator(num_modes, mode=mode).to(device)
    optimizer = optim.Adam(rotator.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        v_pred = rotator(v_src)
        # Minimize MSE between rotated source and targets
        loss = nn.MSELoss()(v_pred, v_tgt)
        loss.backward()
        optimizer.step()
        
    # Evaluate accuracy
    correct = 0
    total = len(valid_pairs)
    
    all_embeds = model.target_embed(torch.arange(len(word2id), device=device))
    
    print(f"\nEvaluating Grammatical Rotation ({name}):")
    with torch.no_grad():
        v_shifted = rotator(v_src)
        for i in range(total):
            src_id = src_ids[i].item()
            tgt_id = tgt_ids[i].item()
            pred_id, sims = find_nearest_neighbor(v_shifted[i], all_embeds, exclude_ids=[src_id])
            
            if pred_id == tgt_id:
                correct += 1
            if i < 4:
                print(f"  Rotated({id2word[src_id]}) -> {id2word[pred_id]} (Target: {id2word[tgt_id]}) [Sim: {sims[tgt_id]:.4f}]")
                
    acc = correct / total if total > 0 else 0.0
    print(f"  Rotation Retrieval Accuracy: {acc:.4f} ({correct}/{total})")
    
    # Print learned angles
    if mode == 'shared':
        print(f"  Learned Phase Shift (theta): {rotator.theta.item():.4f} rad")
    else:
        print(f"  Learned Phase Shifts (theta_n) [first 5 modes]: {rotator.theta[:5].cpu().detach().numpy()}")
        
    return acc

# =============================================================================
# PHRASE COMPOSITIONALITY VERIFICATION
# =============================================================================

def evaluate_compositionality(model, word2id, num_modes):
    """
    Computes convolved phrase representations and prints cosine similarity comparisons.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Test cases: (Word1, Word2) to convolve
    phrases = {
        "red_apple": ("red", "apple"),
        "sweet_fruit": ("sweet", "fruit"),
        "green_apple": ("green", "apple"),
        "blue_car": ("blue", "car"),
        "fast_vehicle": ("fast", "vehicle"),
        "cold_water": ("cold", "water")
    }
    
    # Verify all words are in vocab
    embeds = {}
    for key, (w1, w2) in phrases.items():
        if w1 in word2id and w2 in word2id:
            with torch.no_grad():
                v1 = model.target_embed(torch.tensor(word2id[w1], device=device))
                v2 = model.target_embed(torch.tensor(word2id[w2], device=device))
                # Convolve
                v_conv = compose_circular_convolution(v1, v2, num_modes)
                embeds[key] = v_conv
                
    if len(embeds) < 4:
        print("\nSkipping compositionality: Not enough words in vocabulary.")
        return
        
    def cos_sim(vA, vB):
        return torch.dot(vA, vB) / (torch.norm(vA) * torch.norm(vB) + 1e-8)
        
    print("\nEvaluating Semantic Compositionality via Circular Convolution:")
    # Compare similarities
    comparisons = [
        ("red_apple", "green_apple", "Same noun, similar color category"),
        ("red_apple", "sweet_fruit", "Semantic equivalence (apple is a sweet fruit)"),
        ("blue_car", "fast_vehicle", "Semantic equivalence (car is a fast vehicle)"),
        ("red_apple", "blue_car", "Completely different categories"),
        ("cold_water", "fast_vehicle", "Completely different categories"),
    ]
    
    for k1, k2, desc in comparisons:
        if k1 in embeds and k2 in embeds:
            sim = cos_sim(embeds[k1], embeds[k2]).item()
            print(f"  Sim({k1}, {k2}) = {sim:7.4f} | Description: {desc}")

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=int, default=16, help="Number of Fourier modes")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--reg_weight", type=float, default=0.02, help="Weight decay for high frequencies")
    parser.add_argument("--reg_power", type=float, default=1.5, help="High-frequency penalty power")
    parser.add_argument("--phase_mode", type=str, default="independent", choices=["shared", "independent"], help="Grammar shift angle mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")
    
    # 1. Generate corpus
    sentences, plurals, gender, tenses, capitals = get_grammatical_dataset()
    dataset = TextDataset(sentences, window_size=3, num_negatives=5)
    print(f"Vocabulary Size: {dataset.vocab_size} words")
    print(f"Total Skip-gram Pairs: {len(dataset.pairs)}")
    
    # 2. Define model
    model = SpectralSGNS(dataset.vocab_size, args.modes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Register frequency weights for regularization
    # decay penalty = n^p
    harmonic_n = torch.arange(1, args.modes + 1, dtype=torch.float32, device=device)
    reg_weights = harmonic_n ** args.reg_power
    
    # 3. Training loop
    print("\nTraining Spectral Word Embeddings...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_batches = 0
        
        for targets, contexts, negatives in dataset.get_batches(args.batch_size):
            targets = targets.to(device)
            contexts = contexts.to(device)
            negatives = negatives.to(device)
            
            optimizer.zero_grad()
            pos_scores, neg_scores = model(targets, contexts, negatives)
            
            # SGNS loss: -log(sigmoid(pos)) - sum(log(sigmoid(-neg)))
            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()
            neg_loss = -torch.log(torch.sigmoid(-neg_scores) + 1e-8).sum(dim=-1).mean()
            loss = pos_loss + neg_loss
            
            # Spectral Decay Regularization
            # Penalize higher-frequency coefficients (a_n, b_n) of the target embeddings
            _, an, bn = model.target_embed.get_raw_coefficients(targets)
            # an: (B, N); bn: (B, N)
            spectral_energy = (an**2 + bn**2) * reg_weights.unsqueeze(0)
            reg_loss = args.reg_weight * spectral_energy.mean()
            
            total_loss_with_reg = loss + reg_loss
            total_loss_with_reg.backward()
            optimizer.step()
            
            total_loss += total_loss_with_reg.item()
            total_batches += 1
            
        if epoch <= 3 or epoch % 20 == 0 or epoch == args.epochs:
            avg_loss = total_loss / total_batches
            print(f"  Epoch {epoch:03d} | Avg Loss (incl. Reg): {avg_loss:.4f} | Scale: {model.target_embed.scale.item():.4f}")
            
    # 4. Evaluation: Analogies
    # Formulate analogies list
    # king:man::queen:woman -> pair1 = (man, woman), pair2 = (king, queen)
    gender_analogy_pairs = [
        (("man", "woman"), ("king", "queen")),
        (("boy", "girl"), ("prince", "princess")),
        (("father", "mother"), ("uncle", "aunt")),
        (("brother", "sister"), ("son", "daughter")),
        (("husband", "wife"), ("king", "queen"))
    ]
    
    capital_analogy_pairs = [
        (("paris", "france"), ("london", "uk")),
        (("rome", "italy"), ("tokyo", "japan")),
        (("berlin", "germany"), ("madrid", "spain")),
        (("beijing", "china"), ("delhi", "india")),
        (("ottawa", "canada"), ("cairo", "egypt"))
    ]
    
    evaluate_analogies(model, dataset.word2id, dataset.id2word, gender_analogy_pairs, "Gender")
    evaluate_analogies(model, dataset.word2id, dataset.id2word, capital_analogy_pairs, "Capitals/Countries")
    
    # 5. Evaluation: Grammatical Rotation
    train_grammar_rotation(model, dataset.word2id, dataset.id2word, plurals, "Singular to Plural", epochs=40, mode=args.phase_mode)
    train_grammar_rotation(model, dataset.word2id, dataset.id2word, tenses, "Present to Past Tense", epochs=40, mode=args.phase_mode)
    
    # 6. Evaluation: Compositionality
    evaluate_compositionality(model, dataset.word2id, args.modes)
    
    # 7. Visualization: Wave profiles
    print("\nGenerating Word Wave Profiles Visualization...")
    model.eval()
    t = torch.linspace(0.0, float(model.target_embed.scale.abs().item()), steps=500, device=device)
    
    words_to_plot = ["king", "queen", "man", "woman", "dog", "dogs"]
    # Verify they exist
    plotted_words = [w for w in words_to_plot if w in dataset.word2id]
    
    if len(plotted_words) > 0:
        word_ids = torch.tensor([dataset.word2id[w] for w in plotted_words], dtype=torch.long, device=device)
        with torch.no_grad():
            f_waves = model.target_embed.reconstruct_function(word_ids, t).cpu().numpy()
            
        t_np = t.cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Semantic Waves (king, queen, man, woman)
        plt.subplot(2, 1, 1)
        styles = {
            "king": ("tab:blue", "-"),
            "queen": ("tab:red", "-"),
            "man": ("tab:blue", "--"),
            "woman": ("tab:red", "--")
        }
        for i, word in enumerate(plotted_words):
            if word in styles:
                color, linestyle = styles[word]
                plt.plot(t_np, f_waves[i], label=word, color=color, linestyle=linestyle, linewidth=2)
                
        plt.title("Semantic Wave Profiles (L2-reconstructed functions f(t))")
        plt.xlabel("Domain coordinate t")
        plt.ylabel("Function Value f(t)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Grammatical Shift (dog vs dogs)
        plt.subplot(2, 1, 2)
        for i, word in enumerate(plotted_words):
            if word in ["dog", "dogs"]:
                color = "tab:green" if word == "dog" else "tab:orange"
                plt.plot(t_np, f_waves[i], label=word, color=color, linewidth=2)
                
        plt.title("Grammatical Transformation: Singular vs Plural Waves")
        plt.xlabel("Domain coordinate t")
        plt.ylabel("Function Value f(t)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("fourier_word_waves.png", dpi=150)
        print("Saved wave profiles plot to fourier_word_waves.png")
        plt.close()

if __name__ == "__main__":
    main()
