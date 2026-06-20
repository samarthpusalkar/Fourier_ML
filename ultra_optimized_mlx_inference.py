import os
import math
import argparse
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

# =============================================================================
# HYBRID CONTINUOUS FOURIER ARCHITECTURE (PURE MLX PORT)
# Apple Silicon highly optimized native translation of the PyTorch operations
# =============================================================================

class LinearFourierMixer(nn.Module):
    def __init__(self, channels, num_modes=128, num_heads=12):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        fractional = mx.exp(mx.linspace(math.log(0.0001), 0, num_modes))
        integers = mx.arange(1, num_modes + 1, dtype=mx.float32)
        freq_bands = mx.concatenate([fractional, integers])
        
        self.num_modes = freq_bands.shape[0]
        self.frequencies = freq_bands
        mx.eval(self.frequencies)

        self.q_proj = nn.Linear(channels, self.num_heads * self.num_modes)
        self.k_proj = nn.Linear(channels, self.num_heads * self.num_modes)
        self.v_proj = nn.Linear(channels, channels)
        
        self.proj_v2 = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.norm_in = nn.LayerNorm(channels)

    def __call__(self, x):
        B, seq_len, C = x.shape
        
        norm_x = self.norm_in(x)
        
        Q = nn.elu(self.q_proj(norm_x)) + 1.0
        Q = Q.reshape(B, seq_len, self.num_heads, self.num_modes)
        
        K = nn.elu(self.k_proj(norm_x)) + 1.0
        K = K.reshape(B, seq_len, self.num_heads, self.num_modes)
        
        v1 = self.v_proj(norm_x)
        v2 = nn.silu(self.proj_v2(norm_x))
        
        t = mx.arange(seq_len, dtype=x.dtype) / 32.0
        t = mx.expand_dims(t, 1)
        
        omega_t = 2 * math.pi * t * mx.expand_dims(self.frequencies, 0)
       
        U = mx.expand_dims(mx.expand_dims(mx.cos(omega_t), 0), 2)
        V = mx.expand_dims(mx.expand_dims(mx.sin(omega_t), 0), 2)
        
        Q_cos = Q * U
        Q_sin = Q * V
        K_cos = K * U
        K_sin = K * V
        
        Q_rot = mx.concatenate([Q_cos, Q_sin], axis=-1)
        K_rot = mx.concatenate([K_cos, K_sin], axis=-1)
        
        Q_rot_h = mx.transpose(Q_rot, (0, 2, 1, 3)) # B, H, S, M
        K_rot_h = mx.transpose(K_rot, (0, 2, 3, 1)) # B, H, M, S
        
        A = mx.matmul(Q_rot_h, K_rot_h)
        A = A / (math.sqrt(self.num_modes * 2) * seq_len)
        
        coords = mx.arange(seq_len)
        causal_mask = mx.expand_dims(coords, 1) >= mx.expand_dims(coords, 0)
        A = A * mx.expand_dims(mx.expand_dims(causal_mask, 0), 0)
        
        v1_heads = v1.reshape(B, seq_len, self.num_heads, self.head_dim)
        v1_heads_h = mx.transpose(v1_heads, (0, 2, 1, 3))
        
        v1_token_mixed = mx.matmul(A, v1_heads_h)
        v1_token_mixed = mx.transpose(v1_token_mixed, (0, 2, 1, 3)).reshape(B, seq_len, C)
        
        v3 = v1 + v1_token_mixed * v2
        return self.out_proj(v3) + x


class SoftmaxFourierMixer(nn.Module):
    def __init__(self, channels, num_modes=128, num_heads=12):
        super().__init__()
        self.channels = channels
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        fractional = mx.exp(mx.linspace(math.log(0.0001), 0, num_modes))
        integers = mx.arange(1, num_modes + 1, dtype=mx.float32)
        freq_bands = mx.concatenate([fractional, integers])
        
        self.num_modes = freq_bands.shape[0]
        self.frequencies = freq_bands
        mx.eval(self.frequencies)

        self.q_proj = nn.Linear(channels, self.num_heads * self.num_modes)
        self.k_proj = nn.Linear(channels, self.num_heads * self.num_modes)
        self.v_proj = nn.Linear(channels, channels)
        
        self.proj_v2 = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.norm_in = nn.LayerNorm(channels)

    def __call__(self, x):
        B, seq_len, C = x.shape
        
        norm_x = self.norm_in(x)
        
        Q = self.q_proj(norm_x).reshape(B, seq_len, self.num_heads, self.num_modes)
        K = self.k_proj(norm_x).reshape(B, seq_len, self.num_heads, self.num_modes)
        
        v1 = self.v_proj(norm_x)
        v2 = nn.silu(self.proj_v2(norm_x))
        
        t = mx.arange(seq_len, dtype=x.dtype) / 32.0
        t = mx.expand_dims(t, 1)
        omega_t = 2 * math.pi * t * mx.expand_dims(self.frequencies, 0) 
       
        U = mx.expand_dims(mx.expand_dims(mx.cos(omega_t), 0), 2)
        V = mx.expand_dims(mx.expand_dims(mx.sin(omega_t), 0), 2)
        
        Q_cos = Q * U  
        Q_sin = Q * V
        K_cos = K * U
        K_sin = K * V
        
        Q_rot = mx.concatenate([Q_cos, Q_sin], axis=-1)
        K_rot = mx.concatenate([K_cos, K_sin], axis=-1)
        
        Q_rot_h = mx.transpose(Q_rot, (0, 2, 1, 3)) # B, H, S, M
        K_rot_h = mx.transpose(K_rot, (0, 2, 3, 1)) # B, H, M, S
        
        scores = mx.matmul(Q_rot_h, K_rot_h) / math.sqrt(self.num_modes * 2)
        
        coords = mx.arange(seq_len)
        mask = mx.expand_dims(coords, 1) < mx.expand_dims(coords, 0)
        scores = mx.where(mask, mx.array(-1e9, dtype=scores.dtype), scores)
        
        A = mx.softmax(scores, axis=-1)
        
        v1_heads = v1.reshape(B, seq_len, self.num_heads, self.head_dim)
        v1_heads_h = mx.transpose(v1_heads, (0, 2, 1, 3))
        
        v1_token_mixed = mx.matmul(A, v1_heads_h)
        v1_token_mixed = mx.transpose(v1_token_mixed, (0, 2, 1, 3)).reshape(B, seq_len, C)
        
        v3 = v1 + v1_token_mixed * v2
        return self.out_proj(v3) + x


class HybridSpectralBlock(nn.Module):
    def __init__(self, latent_dim, num_modes=128, is_softmax=False):
        super().__init__()
        if is_softmax:
            self.mixer = SoftmaxFourierMixer(latent_dim, num_modes)
        else:
            self.mixer = LinearFourierMixer(latent_dim, num_modes)
            
        self.ffn = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(0.05)
        )
        
    def __call__(self, x): 
        z = self.mixer(x)
        return z + self.ffn(z)


class HybridFourierLM(nn.Module):
    def __init__(self, vocab_size, latent_dim=768, num_layers=12, num_modes=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, latent_dim)
        
        self.mixers = [
            HybridSpectralBlock(latent_dim, num_modes, is_softmax=(i % 4 == 3)) 
            for i in range(num_layers)
        ]
        
        self.ln_f = nn.LayerNorm(latent_dim)
        self.lm_head = nn.Linear(latent_dim, vocab_size, bias=False)
        
    def __call__(self, input_ids):
        z = self.embedding(input_ids)
        for mixer in self.mixers: 
            z = mixer(z)
        logits = self.lm_head(self.ln_f(z))
        return logits

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = mx.array([input_ids])
    
    print(f"\n[Prompt]: {prompt}")
    print("[Generating...]", end=" ", flush=True)
    
    generated = []
    
    for _ in range(max_new_tokens):
        logits = model(x)
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
        
        next_token = mx.random.categorical(next_token_logits)
        mx.eval(next_token)
        token_id = next_token.item()
        generated.append(token_id)
        
        x = mx.concatenate([x, mx.array([[token_id]])], axis=1)
        mx.eval(x)
        
    final_string = tokenizer.decode(input_ids + generated)
    print(f"\n[Final Output]:\n{final_string}\n")
    return final_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="CodeIsAbstract/Hybrid_test_model_kaggle",
                        help="Hugging Face repo ID to fetch the model from.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional: Path to a local safetensors file. If set, overrides repo_id.")
    parser.add_argument("--max_tokens", type=int, default=150)
    args = parser.parse_args()

    # Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    print("Initializing Native MLX Hybrid Fourier Model...")
    model = HybridFourierLM(vocab_size=len(tokenizer), latent_dim=768, num_layers=12, num_modes=128)
    
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        print(f"Fetching latest checkpoint from Hugging Face Repo: {args.repo_id}...")
        try:
            from huggingface_hub import snapshot_download
            # Get the exact last-checkpoint folder contents
            ckpt_dir = snapshot_download(repo_id=args.repo_id, allow_patterns=["last-checkpoint/*"])
            checkpoint_path = os.path.join(ckpt_dir, "last-checkpoint", "model.safetensors")
            if not os.path.exists(checkpoint_path):
                print(f"[!] Could not find model.safetensors inside last-checkpoint!")
                checkpoint_path = None
            else:
                print(f"Downloaded securely to: {checkpoint_path}")
        except Exception as e:
            print(f"\n[!] Error fetching from HF Hub: {e}")
            checkpoint_path = None

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading native MLX weights from {checkpoint_path}...")
        weights = mx.load(checkpoint_path)
        new_weights = []
        
        for k, v in weights.items():
            if k.startswith("model."):
                k = k[6:] 
                
            # Map PyTorch `mixers.0.ffn.0` to MLX `mixers.0.ffn.layers.0`
            if ".ffn." in k:
                k = k.replace(".ffn.0.", ".ffn.layers.0.")
                k = k.replace(".ffn.1.", ".ffn.layers.1.")
                k = k.replace(".ffn.3.", ".ffn.layers.3.")
                
            new_weights.append((k, v))
            
        model.load_weights(new_weights, strict=False)
        print("✅ Weights successfully mapped to Apple Silicon Unified Memory!")
    else:
        print(f"⚠️ Warning: Could not find any valid checkpoint. Generating with random weights.")
        
    mx.eval(model.parameters())
    
    test_prompts = [
        "Artificial intelligence and Fourier transforms combine to",
        "The history of machine learning algorithms",
        "Deep Neural Networks process language by"
    ]
    
    for prompt in test_prompts:
        generate_text(model, tokenizer, prompt, max_new_tokens=args.max_tokens, temperature=0.7)
