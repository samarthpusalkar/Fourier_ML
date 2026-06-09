"""
tied_spectral_language_model.py — Tied-Weight Massive Scale Spectral LM
======================================================================
Implements Weight-Tying to remove the redundant 15M parameter classification
matrix, freeing up the budget to massively expand the inner spectral engine.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from spectral_core import CoefficientFourierHead

# =============================================================================
# V8 ARCHITECTURE COMPONENTS
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


class GenericSpectralMixer1D(nn.Module):
    def __init__(self, channels, spatial_shape):
        super().__init__()
        self.channels = channels
        self.spatial_shape = (spatial_shape,) if isinstance(spatial_shape, int) else spatial_shape
        self.freq_shape = (self.spatial_shape[0] // 2) + 1
        
        self.gain_real = nn.Parameter(torch.ones(channels, self.freq_shape) * 0.5)
        self.gain_imag = nn.Parameter(torch.zeros(channels, self.freq_shape))
        self.norm = nn.LayerNorm(channels)
        self.activation = LearnableSquareWave()
        self.dropout = nn.Dropout(0.05) 

    def forward(self, x):
        x_fft = torch.fft.rfft(x, dim=1) # (B, Freq, C)
        gain = torch.view_as_complex(torch.stack([self.gain_real, self.gain_imag], dim=-1))
        gain = gain.permute(1, 0)
        x_filtered = x_fft * gain.unsqueeze(0)
        x_out = torch.fft.irfft(x_filtered, n=self.spatial_shape[0], dim=1)
        
        x_out = self.activation(x_out)
        x_out = self.dropout(x_out)
        return self.norm(x_out + x)


class LargeSpectralLM(nn.Module):
    def __init__(self, vocab_size, seq_length=64, latent_dim=512, num_modes=128, 
                 num_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        
        # This embedding matrix will be tied to the output classifier
        self.embedding = nn.Embedding(vocab_size, latent_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(seq_length, latent_dim)
        
        self.mixers = nn.ModuleList([
            GenericSpectralMixer1D(latent_dim, seq_length)
            for _ in range(num_layers)
        ])
        
        self.fourier_head = CoefficientFourierHead(latent_dim, num_modes, init_scale=2.0, grid_type="nufft")
        coeff_dim = 1 + 2 * num_modes
        
        # Project from Fourier coefficients back to the tied latent dimension
        self.pre_classifier = nn.Sequential(
            nn.Linear(coeff_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(0.05)
        )
        
    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        z = self.embedding(x) + self.pos_embedding(pos)
        
        for mixer in self.mixers:
            z = mixer(z)
            
        z_flat = z.view(B * L, self.latent_dim)
        coeffs, _ = self.fourier_head(z_flat)
        pre_logits = self.pre_classifier(coeffs)
        
        # WEIGHT TYING: Use the embedding weights for the final projection
        logits_flat = F.linear(pre_logits, self.embedding.weight)
        
        return logits_flat.view(B, L, self.vocab_size), coeffs

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================================
# DATA PIPELINE
# =============================================================================

def get_dataloaders(dataset_name="wikitext-2-raw-v1", seq_length=64, batch_size=64):
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
    
    train_loader = DataLoader(lm_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    val_loader = DataLoader(lm_datasets["validation"], batch_size=batch_size, collate_fn=data_collator)
    
    return train_loader, val_loader, len(tokenizer)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext-2-raw-v1", choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64) # Reduced batch size for 512 latent dim
    parser.add_argument("--lr", type=float, default=1e-3, help="Base LR for AdamW")
    parser.add_argument("--latent_dim", type=int, default=512) # Expanded latent dim
    parser.add_argument("--num_modes", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=6) # Expanded layers
    parser.add_argument("--seq_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, vocab_size = get_dataloaders(args.dataset, args.seq_length, args.batch_size)
    
    model = LargeSpectralLM(
        vocab_size=vocab_size, 
        seq_length=args.seq_length,
        latent_dim=args.latent_dim, 
        num_modes=args.num_modes, 
        num_layers=args.num_layers
    ).to(device)
    
    print("\n" + "="*60)
    print(f"TIED SPECTRAL LM ({args.dataset})")
    print("="*60)
    print(f"Total Parameters: {count_params(model):,}")
    print(f"Vocab Size: {vocab_size}")
    print(f"Optimizer: AdamW | LR: {args.lr}")
    print("="*60 + "\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_loss = 999.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total_masked = 0.0, 0, 0
        
        for batch in train_loader:
            xb = batch["input_ids"].to(device)
            yb = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits, coeffs = model(xb)
            
            logits_flat = logits.view(-1, vocab_size)
            yb_flat = yb.view(-1)
            
            loss = criterion(logits_flat, yb_flat)
            
            a_n = coeffs[:, 1:1+args.num_modes]
            b_n = coeffs[:, 1+args.num_modes:]
            reg_loss = 1e-4 * (a_n**2 + b_n**2).mean()
            
            (loss + reg_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            mask = yb_flat != -100
            if mask.sum() > 0:
                preds = logits_flat.argmax(dim=-1)
                total_correct += (preds[mask] == yb_flat[mask]).sum().item()
                total_masked += mask.sum().item()
                
        avg_loss = total_loss / len(train_loader)
        acc = (total_correct / total_masked) * 100 if total_masked > 0 else 0
        
        scheduler.step()
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Validation evaluation
        model.eval()
        val_loss, val_correct, val_masked = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                xb = batch["input_ids"].to(device)
                yb = batch["labels"].to(device)
                logits, _ = model(xb)
                logits_flat = logits.view(-1, vocab_size)
                yb_flat = yb.view(-1)
                loss = criterion(logits_flat, yb_flat)
                val_loss += loss.item()
                mask = yb_flat != -100
                if mask.sum() > 0:
                    preds = logits_flat.argmax(dim=-1)
                    val_correct += (preds[mask] == yb_flat[mask]).sum().item()
                    val_masked += mask.sum().item()
                    
        v_loss = val_loss / len(val_loader)
        v_acc = (val_correct / val_masked) * 100 if val_masked > 0 else 0
        print(f"    -> [Val] Loss: {v_loss:.4f} | Acc: {v_acc:.2f}%")
        
        if v_loss < best_loss:
            best_loss = v_loss
            os.makedirs("results", exist_ok=True)
            torch.save(model.state_dict(), "results/best_tied_spectral.pt")

    print(f"\nFinal Best Val Loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
