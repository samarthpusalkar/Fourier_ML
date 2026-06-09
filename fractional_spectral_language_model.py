"""
fractional_spectral_language_model.py — Fractional Frequency Spectral LM
========================================================================
Implements a custom FractionalSpectralMixer1D to process sub-integer
frequencies (resolution < 1) in the hidden layers, bypassing the rigid
integer spacing of standard FFT to preserve long-wavelength topology.
Massively scaled to 2048 dimensions with Weight Tying.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import math

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.modeling_outputs import MaskedLMOutput
from torch.utils.data import DataLoader
from spectral_core import CoefficientFourierHead

# =============================================================================
# FRACTIONAL ARCHITECTURE COMPONENTS
# =============================================================================

class LearnableSquareWave(nn.Module):
    """y = A * tanh(steepness * sin(freq * x + phase))"""
    def __init__(self, steepness=10.0):
        super().__init__()
        self.steepness = steepness
        self.A = nn.Parameter(torch.ones(1))
        self.freq = nn.Parameter(torch.ones(1))
        self.phase = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        return self.A * torch.tanh(self.steepness * torch.sin(self.freq * x + self.phase))


class FractionalSpectralMixer1D(nn.Module):
    """
    Evaluates frequencies at custom resolution (e.g. 1/20) instead of integer harmonics.
    Projects the sequence onto a custom cosine/sine basis, applies complex gain,
    and projects back via the transpose basis (acting as a generalized fractional filter).
    """
    def __init__(self, channels, seq_length, num_modes=33, resolution=0.05):
        super().__init__()
        self.channels = channels
        self.seq_length = seq_length
        self.num_modes = num_modes
        self.resolution = resolution
        
        # Frequencies: Non-uniform Gaussian-spaced distribution
        # Dense near zero to capture topological semantics, but bounded to prevent explosion.
        quantiles = torch.linspace(0.0, 0.99, num_modes, dtype=torch.float32)
        freqs = torch.erfinv(quantiles) * resolution
        
        # Time steps: [0, 1, ..., seq_length-1]
        t = torch.arange(seq_length, dtype=torch.float32)
        
        # Basis matrix: shape (seq_length, num_modes)
        # We use 2*pi * f * t / seq_length so that when resolution=1, it exactly matches standard FFT.
        arg = 2 * np.pi * t.unsqueeze(1) * freqs.unsqueeze(0) / seq_length
        cos_basis = torch.cos(arg) # (L, modes)
        sin_basis = torch.sin(arg) # (L, modes)
        
        self.register_buffer('cos_basis', cos_basis)
        self.register_buffer('sin_basis', sin_basis)
        
        # Learnable complex gain for the fractional frequencies
        # Scaled initialization by 1/sqrt(channels) to prevent variance explosion
        stdv = 1.0 / math.sqrt(channels)
        self.gain_real = nn.Parameter(torch.randn(channels, num_modes) * stdv)
        self.gain_imag = nn.Parameter(torch.randn(channels, num_modes) * stdv)
        
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        # x: (B, L, C)
        # MPS Optimization: Replace einsum with highly optimized batched matmul (GEMM)
        
        # 1. Permute x to (Batch, Channels, Length)
        x_c = x.permute(0, 2, 1)
        
        # 2. Transform to Fractional Frequency Domain
        # (B, C, L) @ (L, modes) -> (B, C, modes)
        X_real = torch.matmul(x_c, self.cos_basis)
        X_imag = torch.matmul(x_c, self.sin_basis)
        
        # 3. Apply complex gain
        # gain_real/imag are (Channels, modes), so we unsqueeze for batch broadcast
        G_r = self.gain_real.unsqueeze(0) # (1, C, modes)
        G_i = self.gain_imag.unsqueeze(0) # (1, C, modes)
        
        Y_real = X_real * G_r - X_imag * G_i
        Y_imag = X_real * G_i + X_imag * G_r
        
        # 4. Transform back to time domain
        # (B, C, modes) @ (modes, L) -> (B, C, L)
        y_c = torch.matmul(Y_real, self.cos_basis.T) + torch.matmul(Y_imag, self.sin_basis.T)
            
        # 5. Normalize and permute back to (B, L, C)
        scale_factor = math.sqrt(2.0 / self.seq_length)
        y = y_c.permute(0, 2, 1) * scale_factor
        
        x_out = self.activation(y)
        x_out = self.dropout(x_out)
        return self.norm(x_out + x)


class SpectralBlock(nn.Module):
    def __init__(self, latent_dim, seq_length, num_modes=33, resolution=0.05):
        super().__init__()
        # Spatial Mixing (Your Fourier Machinery)
        self.mixer = FractionalSpectralMixer1D(latent_dim, seq_length, num_modes, resolution)
        
        # Channel Mixing (MLP)
        self.ffn = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(0.05)
        )
        
    def forward(self, x):
        x = self.mixer(x)
        x = x + self.ffn(x) # Mix features across channels
        return x


class LargeSpectralLM(nn.Module):
    def __init__(self, vocab_size, seq_length=64, latent_dim=2048, num_modes=128, 
                 num_layers=7, resolution=0.05):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        
        # Tied Embedding Matrix
        self.embedding = nn.Embedding(vocab_size, latent_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(seq_length, latent_dim)
        
        # Fractional Mixers instead of integer FFT Mixers
        # Using seq_length//2 + 1 modes to match standard FFT parameter count, but at 1/20 resolution
        mixer_modes = (seq_length // 2) + 1
        self.mixers = nn.ModuleList([
            SpectralBlock(latent_dim, seq_length, num_modes=mixer_modes, resolution=resolution)
            for _ in range(num_layers)
        ])
        
        # Final LayerNorm before prediction
        self.ln_f = nn.LayerNorm(latent_dim)
        
        # ==========================================
        # CRITICAL FIX: TRANSFORMER WEIGHT INITIALIZATION
        # ==========================================
        # Scale embeddings down to prevent logit explosion at d_model=2048
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        
        # Ensure padding token stays zeroed
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
                
        # Initialize FFN linear layers safely
        for block in self.mixers:
            for module in block.ffn:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(module.bias)
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
        # We assume input_ids is our x
        x = input_ids
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        z = self.embedding(x) + self.pos_embedding(pos)
        
        for mixer in self.mixers:
            z = mixer(z)
            
        z = self.ln_f(z)
        
        # WEIGHT TYING: Use the embedding weights for the final projection
        logits = F.linear(z, self.embedding.weight)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100)
            
        return MaskedLMOutput(loss=loss, logits=logits)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================================
# DATA PIPELINE
# =============================================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    accuracy = (predictions[mask] == labels[mask]).mean()
    return {"accuracy": accuracy}

def get_datasets(dataset_name="wikitext-2-raw-v1", seq_length=64):
    print(f"Loading {dataset_name} from HuggingFace...")
    raw_datasets = load_dataset("wikitext", dataset_name)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"])
        
    print("Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // seq_length) * seq_length
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    print("Grouping into sequences...")
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=4)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    
    return lm_datasets["train"], lm_datasets["validation"], data_collator, len(tokenizer)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext-2-raw-v1", choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=2048) 
    parser.add_argument("--num_modes", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=7) 
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--seq_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    train_dataset, val_dataset, data_collator, vocab_size = get_datasets(args.dataset, args.seq_length)
    
    model = LargeSpectralLM(
        vocab_size=vocab_size, 
        seq_length=args.seq_length,
        latent_dim=args.latent_dim, 
        num_modes=args.num_modes, 
        num_layers=args.num_layers,
        resolution=args.resolution
    )
    
    print("\n" + "="*60)
    print(f"FRACTIONAL SPECTRAL LM ({args.dataset}) via HF TRAINER")
    print("="*60)
    print(f"Total Parameters: {count_params(model):,}")
    print(f"Vocab Size: {vocab_size}")
    print(f"Latent Dim: {args.latent_dim} | Layers: {args.num_layers}")
    print("="*60 + "\n")
    
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        optim="adamw_hf",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        report_to="none" # Disable wandb for local test
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Save the final best weights as well in the generic format
    torch.save(model.state_dict(), "results/best_fractional_spectral.pt")
    print("\nTraining complete! Best weights saved to results/best_fractional_spectral.pt")

if __name__ == "__main__":
    main()
