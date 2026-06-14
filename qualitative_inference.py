import torch
import os
from transformers import AutoTokenizer
import os
import math
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput



class CausalContinuousFourierMixer1D(nn.Module):
    def __init__(self, channels, num_modes=128, num_heads=12):
        super().__init__()
        self.channels = channels
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # ---------------------------------------------------------------------
        # MULTI-HEAD CONTINUOUS FOURIER WEIGHTS
        # Instead of 768 independent channel filters (which causes RAM explosion),
        # we group them into 12 Heads (like standard Attention). This maintains 
        # the exact mathematical freedom while shrinking memory by 64x.
        # ---------------------------------------------------------------------
        self.fourier_amplitudes = nn.Parameter(torch.randn(num_heads, num_modes) / math.sqrt(num_modes))
        self.fourier_phases = nn.Parameter(torch.randn(num_heads, num_modes))
        
        self.register_buffer("frequencies", torch.arange(1, num_modes + 1, dtype=torch.float32))

        self.proj_v1 = nn.Linear(channels, channels)
        self.proj_v2 = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, seq_len, C = x.shape
        
        v1 = self.proj_v1(x)
        v2 = self.activation(self.proj_v2(x))
        
        # Create continuous time grid safely
        # FIX: We MUST use absolute scaling based on the training length (512).
        # linspace(0, 1, seq_len) causes hyper-frequencies on short prompts!
        # During training (seq_len=512), linspace gives steps of 1/511. 
        t = (torch.arange(seq_len, device=x.device, dtype=x.dtype) / 511.0).view(-1, 1)
        
        omega_t = 2 * math.pi * t * self.frequencies.unsqueeze(0) # (seq_len, num_modes)
        U = torch.cos(omega_t) # (seq_len, num_modes)
        V = torch.sin(omega_t) # (seq_len, num_modes)
        
        W_cos = self.fourier_amplitudes * torch.cos(self.fourier_phases) # (num_heads, num_modes)
        W_sin = self.fourier_amplitudes * torch.sin(self.fourier_phases) # (num_heads, num_modes)
        
        # =====================================================================
        # THE FINAL XLA HARDWARE FIX (Pure Matrix Multiplication)
        # We replace all 'einsum' calls with explicit, standard 'matmul' and 
        # '.contiguous()' reshaping. XLA sometimes assigns broken memory strides 
        # (tensor_data crash) when handling heavily interleaved einsum operations.
        # =====================================================================
        
        # 1. Generate Query Matrices (Q) by rotating the weights through time
        # Explicit .expand() and .contiguous() is crucial for XLA to avoid zero-stride bugs
        U_exp = U.unsqueeze(1).expand(-1, self.num_heads, -1).contiguous() # (seq_len, num_heads, num_modes)
        V_exp = V.unsqueeze(1).expand(-1, self.num_heads, -1).contiguous()
        
        W_cos_exp = W_cos.unsqueeze(0).expand(seq_len, -1, -1).contiguous() # (seq_len, num_heads, num_modes)
        W_sin_exp = W_sin.unsqueeze(0).expand(seq_len, -1, -1).contiguous()
        
        P_cos = U_exp * W_cos_exp + V_exp * W_sin_exp # (seq_len, num_heads, num_modes)
        P_sin = V_exp * W_cos_exp - U_exp * W_sin_exp # (seq_len, num_heads, num_modes)
        
        # Transpose for batched MatMul: (num_heads, seq_len, num_modes)
        P_cos_h = P_cos.transpose(0, 1)
        P_sin_h = P_sin.transpose(0, 1)
        
        # 2. Compute the exact Toeplitz Matrix via Q * K^T
        # K_matrix[h, t, j] = P[h, t, m] @ U.T[m, j]
        M_cos = torch.matmul(P_cos_h, U.T) # (num_heads, seq_len, seq_len)
        M_sin = torch.matmul(P_sin_h, V.T)
        K_matrix = M_cos + M_sin 
        
        # 3. Apply standard Causal Mask safely (Explicitly matching dtype to prevent FP32 upcast)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype))
        K_matrix = K_matrix * causal_mask.unsqueeze(0)
        
        # 4. Route Values (v1) using standard Attention Batched MatMul
        # v1: (B, seq_len, num_heads, head_dim) -> (B, num_heads, seq_len, head_dim)
        v1_heads = v1.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K_expanded: (B, num_heads, seq_len, seq_len)
        # Explicitly expanding the batch dimension for XLA matmul safety
        K_expanded = K_matrix.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        
        # MatMul: (1, H, seq_len, seq_len) @ (B, H, seq_len, D) -> (B, H, seq_len, D)
        v1_token_mixed = torch.matmul(K_expanded, v1_heads)
        
        # 5. Flatten back to standard format safely
        # Explicit contiguous() prevents XLA memory stride bugs (tensor_data crash)
        v1_token_mixed = v1_token_mixed.transpose(1, 2).contiguous().view(B, seq_len, C)
        
        # Scale to prevent activation variance explosion
        v1_token_mixed = v1_token_mixed / math.sqrt(seq_len)
        
        # Vector mixing
        v3 = v1_token_mixed * v2
        
        return self.norm(self.out_proj(v3)) + x

class CausalSpectralBlock(nn.Module):
    def __init__(self, latent_dim, num_modes=128):
        super().__init__()
        self.mixer = CausalContinuousFourierMixer1D(latent_dim, num_modes)
        self.ffn = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(0.05)
        )
        
    def forward(self, x): 
        z = self.mixer(x)
        return z + self.ffn(z)

class ContinuousFourierLM(nn.Module):
    def __init__(self, vocab_size, latent_dim=768, num_layers=12, num_modes=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, latent_dim, padding_idx=0)
        
        self.mixers = nn.ModuleList([
            CausalSpectralBlock(latent_dim, num_modes) 
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(latent_dim)
        
        self.lm_head = nn.Linear(latent_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights to standard Transformer variance
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        
    def forward(self, input_ids, labels=None, **kwargs):
        z = self.embedding(input_ids)
        for mixer in self.mixers: 
            z = mixer(z)
        logits = self.lm_head(self.ln_f(z))
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return CausalLMOutput(loss=loss, logits=logits)

class ContinuousFourierLM(nn.Module):
    def __init__(self, vocab_size, latent_dim=768, num_layers=12, num_modes=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, latent_dim, padding_idx=0)
        
        self.mixers = nn.ModuleList([
            CausalSpectralBlock(latent_dim, num_modes) 
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(latent_dim)
        
        self.lm_head = nn.Linear(latent_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights to standard Transformer variance
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        
    def forward(self, input_ids, labels=None, **kwargs):
        z = self.embedding(input_ids)
        for mixer in self.mixers: 
            z = mixer(z)
        logits = self.lm_head(self.ln_f(z))
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return CausalLMOutput(loss=loss, logits=logits)

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    print(f"\n[Prompt]: {prompt}")
    print("[Generating...]")
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(input_ids)
            # Grab the logits for the very last token
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature
            
            # Top-K filtering for better text quality
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
            
            # Sample from the filtered distribution
            next_token_index = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(top_k_indices, -1, next_token_index)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
    return tokenizer.decode(input_ids[0])

if __name__ == "__main__":
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "mps"
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 2. Initialize Model
    print("Initializing Continuous Fourier Model...")
    model = ContinuousFourierLM(vocab_size=len(tokenizer), latent_dim=768, num_layers=12, num_modes=128)
    
    # 3. Load Checkpoint (Replace this path with your actual Kaggle checkpoint path!)
    checkpoint_path = "/Users/samarthpusalkar/Downloads/pytorch_model (1).bin" 
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: Could not find {checkpoint_path}. Generating with random weights!")
        
    model.to(device)
    
    # 4. Run Qualitative Inference Tests
    test_prompts = [
        "The history of the Roman Empire is",
        "Artificial intelligence and machine learning are",
        "The primary function of a database is to"
    ]
    
    for prompt in test_prompts:
        result = generate_text(model, tokenizer, prompt, max_new_tokens=600, temperature=0.7)
        print(f"[Output]: {result}\n")
