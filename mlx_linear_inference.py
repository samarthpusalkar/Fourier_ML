import os
import math
import argparse
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

# =============================================================================
# CONTINUOUS FOURIER ARCHITECTURE (PURE MLX PORT)
# Apple Silicon highly optimized native translation of the PyTorch operations
# =============================================================================

class CausalContinuousFourierMixer1D(nn.Module):
    def __init__(self, channels, num_modes=128, num_heads=12):
        super().__init__()
        self.channels = channels
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # MLX parameter arrays
        self.fourier_amplitudes = mx.random.normal((num_heads, num_modes)) / math.sqrt(num_modes)
        self.fourier_phases = mx.random.normal((num_heads, num_modes))
        
        # Frequencies are explicitly tracked
        freq_bands = mx.power(10.0, mx.linspace(math.log10(0.01), math.log10(128.0), num_modes))
        self.frequencies = freq_bands

        self.proj_v1 = nn.Linear(channels, channels)
        self.proj_v2 = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def __call__(self, x, state=None):
        B, seq_len, C = x.shape
        
        v1 = self.proj_v1(x)
        # MLX natively supports nn.silu directly
        v2 = nn.silu(self.proj_v2(x))
        
        # Create continuous time grid
        current_step = 0 if state is None else state[2]
        t = mx.arange(current_step, current_step + seq_len, dtype=x.dtype) / 511.0
        t = mx.expand_dims(t, 1) # (seq_len, 1)
        
        omega_t = 2 * math.pi * t * mx.expand_dims(self.frequencies, 0) # (seq_len, num_modes)
        U = mx.cos(omega_t) # (seq_len, num_modes)
        V = mx.sin(omega_t) # (seq_len, num_modes)
        
        W_cos = self.fourier_amplitudes * mx.cos(self.fourier_phases) # (num_heads, num_modes)
        W_sin = self.fourier_amplitudes * mx.sin(self.fourier_phases) # (num_heads, num_modes)
        
        # Query rotation
        U_exp = mx.expand_dims(U, 1) # (seq_len, 1, num_modes)
        V_exp = mx.expand_dims(V, 1)
        
        W_cos_exp = mx.expand_dims(W_cos, 0) # (1, num_heads, num_modes)
        W_sin_exp = mx.expand_dims(W_sin, 0)
        
        P_cos = U_exp * W_cos_exp + V_exp * W_sin_exp # (seq_len, num_heads, num_modes)
        P_sin = V_exp * W_cos_exp - U_exp * W_sin_exp # (seq_len, num_heads, num_modes) 
        
        # =====================================================================
        # O(N) LINEAR ATTENTION REFORMULATION
        # =====================================================================
        # Instead of materializing an NxN matrix (M_cos), we use the 
        # associative property to compute (P * U^T) * V as P * (U^T * V).
        # The causal mask simply becomes a cumulative sum over the sequence!
        
        # Value routing shape: (B, seq_len, num_heads, head_dim)
        v1_heads = v1.reshape(B, seq_len, self.num_heads, self.head_dim)
        
        # Reshape frequencies U, V to (1, seq_len, 1, num_modes, 1)
        U_exp_state = mx.expand_dims(mx.expand_dims(mx.expand_dims(U, 0), 2), 4)
        V_exp_state = mx.expand_dims(mx.expand_dims(mx.expand_dims(V, 0), 2), 4)
        
        # Reshape values to (B, seq_len, num_heads, 1, head_dim)
        v1_heads_state = mx.expand_dims(v1_heads, 3)
        
        # Outer product per token. Shape: (B, seq_len, num_heads, num_modes, head_dim)
        U_outer_V = U_exp_state * v1_heads_state
        V_outer_V = V_exp_state * v1_heads_state
        
        # The CAUSAL linear scan: just a cumulative sum over time (axis=1)
        S_cos = mx.cumsum(U_outer_V, axis=1) 
        S_sin = mx.cumsum(V_outer_V, axis=1)
        
        # Add previous state if caching!
        if state is not None:
            prev_S_cos, prev_S_sin, _ = state
            S_cos = S_cos + mx.expand_dims(prev_S_cos, 1)
            S_sin = S_sin + mx.expand_dims(prev_S_sin, 1)
            
        new_state = (S_cos[:, -1, ...], S_sin[:, -1, ...], current_step + seq_len)
        
        # Reshape P to (1, seq_len, num_heads, num_modes, 1)
        P_cos_state = mx.expand_dims(mx.expand_dims(P_cos, 0), 4)
        P_sin_state = mx.expand_dims(mx.expand_dims(P_sin, 0), 4)
        
        # Multiply by P and sum out the modes (axis=3)
        # Shape: (B, seq_len, num_heads, head_dim)
        Y_cos = mx.sum(P_cos_state * S_cos, axis=3)
        Y_sin = mx.sum(P_sin_state * S_sin, axis=3)
        
        # Final output
        v1_token_mixed = Y_cos + Y_sin
        
        # Flatten back
        v1_token_mixed = v1_token_mixed.reshape(B, seq_len, C)
        v1_token_mixed = v1_token_mixed / math.sqrt(seq_len)
        
        v3 = v1_token_mixed * v2
        
        return self.norm(self.out_proj(v3)) + x, new_state

class CausalSpectralBlock(nn.Module):
    def __init__(self, latent_dim, num_modes=128):
        super().__init__()
        self.mixer = CausalContinuousFourierMixer1D(latent_dim, num_modes)
        
        # This matches the PyTorch nn.Sequential structure perfectly,
        # ensuring the Hugging Face weights load directly without key mismatches!
        self.ffn = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(0.05)
        )
        
    def __call__(self, x, state=None): 
        z, new_state = self.mixer(x, state)
        return z + self.ffn(z), new_state

class ContinuousFourierLM(nn.Module):
    def __init__(self, vocab_size, latent_dim=768, num_layers=12, num_modes=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, latent_dim)
        
        # Module lists map cleanly to pure Python lists in MLX
        self.mixers = [CausalSpectralBlock(latent_dim, num_modes) for _ in range(num_layers)]
        
        self.ln_f = nn.LayerNorm(latent_dim)
        self.lm_head = nn.Linear(latent_dim, vocab_size, bias=False)
        
    def __call__(self, input_ids, cache=None):
        z = self.embedding(input_ids)
        new_cache = []
        for i, mixer in enumerate(self.mixers):
            state = None if cache is None else cache[i]
            z, new_state = mixer(z, state)
            new_cache.append(new_state)
        logits = self.lm_head(self.ln_f(z))
        return logits, new_cache

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7):
    # MLX arrays instead of PyTorch Tensors
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = mx.array([input_ids])
    
    print(f"\n[Prompt]: {prompt}")
    print("[Generating...]", end=" ", flush=True)
    
    generated = []
    
    # PREFILL PHASE: Build state token by token!
    cache = None
    for i, t_id in enumerate(input_ids[:-1]):
        x_step = mx.array([[t_id]])
        _, cache = model(x_step, cache=cache)
        # Periodically evaluate to prevent the compute graph from growing infinitely
        if i % 100 == 0:
            mx.eval(*[layer[0] for layer in cache], *[layer[1] for layer in cache])
            
    # Final token of prompt gives first generated token
    x = mx.array([[input_ids[-1]]])
    logits, cache = model(x, cache=cache)
    next_token_logits = logits[:, -1, :] / temperature
    next_token = mx.random.categorical(next_token_logits)
    token_id = next_token.item()
    generated.append(token_id)
    
    # GENERATION PHASE
    x = mx.array([[token_id]])
    for _ in range(max_new_tokens - 1):
        logits, cache = model(x, cache=cache)
        next_token_logits = logits[:, -1, :] / temperature
        
        next_token = mx.random.categorical(next_token_logits)
        token_id = next_token.item()
        generated.append(token_id)
        x = mx.array([[token_id]])
        
    final_string = tokenizer.decode(input_ids + generated)
    print(f"\n[Final Output]:\n{final_string}\n")
    return final_string

def run_context_scaling_test(model, tokenizer, max_new_tokens=50, temperature=0.7):
    print("\n" + "="*80)
    print("🚀 RUNNING CONTEXT CONTINUATION SCALING TEST")
    print("="*80)
    
    # We will use the contents of database_wiki.txt as our long natural text prompt
    try:
        with open("database_wiki.txt", "r", encoding="utf-8") as f:
            full_text = f.read()
    except FileNotFoundError:
        full_text = "The history of machine learning is fascinating. " * 50
        
    # Encode the full text once
    full_input_ids = tokenizer.encode(full_text, add_special_tokens=False)
    # The user wants to start from 600 and scale up to see where it breaks
    test_lengths = [200, 600, 1000, 1500, 2000]
    
    for target_length in test_lengths:
        print(f"\n--- Testing Context Length: {target_length} Tokens ---")
        
        if len(full_input_ids) < target_length:
            print(f"Warning: Not enough text for {target_length} tokens. only contains {len(full_input_ids)} actual tokens Skipping.")
            continue
            
        input_ids = full_input_ids[:target_length]
        
        x = mx.array([input_ids])
        
        generated = []
        print("[Prompt Last 10 words]" + tokenizer.decode(input_ids[-10:]), end=" ", flush=True)
        
        # PREFILL PHASE: Build state token by token!
        cache = None
        for i, t_id in enumerate(input_ids[:-1]):
            x_step = mx.array([[t_id]])
            _, cache = model(x_step, cache=cache)
            if i % 100 == 0:
                mx.eval(*[layer[0] for layer in cache], *[layer[1] for layer in cache])
                
        # Final token of prompt gives first generated token
        x = mx.array([[input_ids[-1]]])
        logits, cache = model(x, cache=cache)
        next_token_logits = logits[:, -1, :] / temperature
        next_token = mx.random.categorical(next_token_logits)
        token_id = next_token.item()
        generated.append(token_id)
        
        # GENERATION PHASE
        x = mx.array([[token_id]])
        for _ in range(max_new_tokens - 1):
            logits, cache = model(x, cache=cache)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = mx.random.categorical(next_token_logits)
            token_id = next_token.item()
            generated.append(token_id)
            x = mx.array([[token_id]])
            
        final_output = tokenizer.decode(generated)
        print(f"\n[Generated Continuation Output]: {final_output}")
        
    print("\n" + "="*80 + "\n")

# =============================================================================
# HUGGING FACE HUB FETCH & INFERENCE RUNNER
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Replace the repo_id with yours once training is pushed!
    parser.add_argument("--repo_id", type=str, default="CodeIsAbstract/Fourier_LM_Checkpoints_Continued",
                        help="Hugging Face repo ID to fetch the model from.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional: Path to a local safetensors file. If set, overrides repo_id.")
    parser.add_argument("--max_tokens", type=int, default=150)
    args = parser.parse_args()

    # 1. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 2. Initialize Model
    print("Initializing Native MLX Continuous Fourier Model...")
    model = ContinuousFourierLM(vocab_size=len(tokenizer), latent_dim=768, num_layers=12, num_modes=128)
    
    # 3. Load Checkpoint directly via Safetensors
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        print(f"Fetching latest checkpoint from Hugging Face Repo: {args.repo_id}...")
        try:
            from huggingface_hub import hf_hub_download
            # Fetches the model directly to your ~/.cache/huggingface safely
            checkpoint_path = hf_hub_download(repo_id=args.repo_id, filename="model.safetensors")
            print(f"Downloaded securely to: {checkpoint_path}")
        except Exception as e:
            print(f"\n[!] Error fetching from HF Hub: {e}")
            print("To push to hub during training, ensure your HF_TOKEN is valid.")
            print("Attempting to look for local fallback instead...\n")
            checkpoint_path = "./CausalFourierLM_Checkpoints_BiggerDataset_GPU/best_model_streaming/model.safetensors"

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading native MLX weights from {checkpoint_path}...")
        
        # Intercept weights to map PyTorch Sequential keys to MLX Sequential keys
        weights = mx.load(checkpoint_path)
        new_weights = []
        
        for k, v in weights.items():
            if k.startswith("model."):
                k = k[6:] # Strip HF wrapper
                
            # Map PyTorch `ffn.0` to MLX `ffn.layers.0`
            if ".ffn." in k:
                k = k.replace(".ffn.0.", ".ffn.layers.0.")
                k = k.replace(".ffn.1.", ".ffn.layers.1.")
                k = k.replace(".ffn.3.", ".ffn.layers.3.")
                
            new_weights.append((k, v))
            
        model.load_weights(new_weights, strict=False)
        print("✅ Weights successfully mapped to Apple Silicon Unified Memory!")
    else:
        print(f"⚠️ Warning: Could not find any checkpoint. Generating with random weights.")
        
    # We explicitly force evaluation to flush the initialization
    mx.eval(model.parameters())
    
    # 4. Run Qualitative Inference (Q&A Style)
    test_prompts = [
        "Q: Why is the sky blue?\nA:",
        "Q: What is the capital of France?\nA:",
        "Q: How does a database store information?\nA:",
        "The history of the Roman Empire is",
    ]
    
    for prompt in test_prompts:
        generate_text(model, tokenizer, prompt, max_new_tokens=args.max_tokens, temperature=0.7)
        
    # 5. Run Context Scaling Test
    # Let's test a natural continuation at 2000 tokens before trying needle in haystack
    run_context_scaling_test(model, tokenizer, max_new_tokens=50, temperature=0.7)
