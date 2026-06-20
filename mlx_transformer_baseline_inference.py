import os
import math
import argparse
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

# =============================================================================
# BASELINE TRANSFORMER ARCHITECTURE (PURE MLX PORT)
# Apple Silicon native translation of the PyTorch baseline operations
# =============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)

    def __call__(self, x):
        B, T, C = x.shape
        
        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scale = math.sqrt(self.head_dim)
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / scale
        
        # Causal mask
        coords = mx.arange(T)
        mask = mx.expand_dims(coords, 1) < mx.expand_dims(coords, 0)
        mask = mask * -1e9
        scores = scores + mask
        
        attn = mx.softmax(scores, axis=-1)
        y = mx.matmul(attn, v)
        
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        return self.c_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        
        # MLP
        m = self.ln_2(x)
        m = nn.gelu(self.c_fc(m))
        m = self.c_proj(m)
        
        x = x + m
        return x

class StandardTransformerLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, num_attention_heads=12, max_position_embeddings=512):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Module lists map cleanly to pure Python lists in MLX
        self.blocks = [TransformerBlock(hidden_size, num_attention_heads) for _ in range(num_layers)]
        
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def __call__(self, input_ids):
        B, T = input_ids.shape
        pos = mx.arange(T)
        
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos)
        
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7):
    # MLX arrays instead of PyTorch Tensors
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = mx.array([input_ids])
    
    print(f"\n[Prompt]: {prompt}")
    print("[Generating...]", end=" ", flush=True)
    
    generated = []
    
    for _ in range(max_new_tokens):
        # We only evaluate the network
        logits = model(x)
        next_token_logits = logits[:, -1, :] / temperature
        
        # MLX native Categorical sampling
        next_token = mx.random.categorical(next_token_logits)
        token_id = next_token.item()
        generated.append(token_id)
        
        # Append to sequence dynamically
        x = mx.concatenate([x, mx.array([[token_id]])], axis=1)
        
    final_string = tokenizer.decode(input_ids + generated)
    print(f"\n[Final Output]:\n{final_string}\n")
    return final_string

# =============================================================================
# HUGGING FACE HUB FETCH & INFERENCE RUNNER
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="CodeIsAbstract/Streaming_Transformers_Baseline_Checkpoints",
                        help="Hugging Face repo ID to fetch the model from.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional: Path to a local safetensors file. If set, overrides repo_id.")
    parser.add_argument("--max_tokens", type=int, default=150)
    args = parser.parse_args()

    # 1. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 2. Initialize Model
    print("Initializing Native MLX Transformer Baseline Model...")
    model = StandardTransformerLM(vocab_size=len(tokenizer), hidden_size=768, num_layers=12, num_attention_heads=12)
    
    # 3. Load Checkpoint directly via Safetensors
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        print(f"Fetching latest checkpoint from Hugging Face Repo: {args.repo_id}...")
        try:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(repo_id=args.repo_id, filename="model.safetensors")
            print(f"Downloaded securely to: {checkpoint_path}")
        except Exception as e:
            print(f"\n[!] Error fetching from HF Hub: {e}")
            print("Attempting to look for local fallback instead...\n")
            checkpoint_path = "./TransformerBaseline_Checkpoints_BiggerDataset_GPU/best_model_streaming/model.safetensors"

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading native MLX weights from {checkpoint_path}...")
        
        # Intercept weights to map PyTorch keys to MLX keys
        weights = mx.load(checkpoint_path)
        new_weights = []
        
        for k, v in weights.items():
            if k.startswith("model."):
                k = k[6:] # Strip HF wrapper
                
            # Map PyTorch `blocks.0` to MLX `blocks.0`
            # In the PyTorch model, we manually unrolled MLP as c_fc and c_proj directly inside TransformerBlock.
            # And attention is MultiHeadAttention.
            
            # The weights map directly since we matched names!
            # The only thing is MLX Embedding weights are just `weight`, matching PyTorch!
            # No special re-mapping needed for MLP since we removed nn.Sequential in PyTorch.
            
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
        "Artificial intelligence and machine learning are",
        "The primary function of a database is to"
    ]
    for prompt in test_prompts:
        generate_text(model, tokenizer, prompt, max_new_tokens=args.max_tokens, temperature=0.1)
