import warnings
warnings.filterwarnings("ignore")

import os
# Mount Google Drive for safe checkpointing
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_PATH = "/content/drive/MyDrive/SpectralLM_Checkpoints"
    os.makedirs(DRIVE_PATH, exist_ok=True)
    print(f"Mounted Drive. Saving checkpoints to {DRIVE_PATH}")
except ImportError:
    print("Not running in Colab. Saving locally.")
    DRIVE_PATH = "./results"

# TPU Detection
try:
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
    print("TPU detected. XLA environment loaded.")
except ImportError:
    TPU_AVAILABLE = False
    print("No TPU detected. Running standard HF Trainer.")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import math
import sys

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.modeling_outputs import MaskedLMOutput

# =============================================================================
# FRACTIONAL ARCHITECTURE COMPONENTS
# =============================================================================

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
        quantiles = torch.linspace(0.0, 0.99, num_modes, dtype=torch.float32)
        freqs = torch.erfinv(quantiles) * resolution
        
        # Time steps: [0, 1, ..., seq_length-1]
        t = torch.arange(seq_length, dtype=torch.float32)
        
        # Basis matrix: shape (seq_length, num_modes)
        arg = 2 * np.pi * t.unsqueeze(1) * freqs.unsqueeze(0) / seq_length
        cos_basis = torch.cos(arg) # (L, modes)
        sin_basis = torch.sin(arg) # (L, modes)
        
        self.register_buffer('cos_basis', cos_basis)
        self.register_buffer('sin_basis', sin_basis)
        
        # Learnable complex gain for the fractional frequencies
        stdv = 1.0 / math.sqrt(channels)
        self.gain_real = nn.Parameter(torch.randn(channels, num_modes) * stdv)
        self.gain_imag = nn.Parameter(torch.randn(channels, num_modes) * stdv)
        
        self.norm = nn.LayerNorm(channels)
        # CRITICAL FIX: SiLU instead of GELU so negative Fourier amplitudes aren't wiped out!
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        # 1. Permute x to (Batch, Channels, Length)
        x_c = x.permute(0, 2, 1)
        
        # 2. Transform to Fractional Frequency Domain
        X_real = torch.matmul(x_c, self.cos_basis)
        X_imag = torch.matmul(x_c, self.sin_basis)
        
        # 3. Apply complex gain
        G_r = self.gain_real.unsqueeze(0) # (1, C, modes)
        G_i = self.gain_imag.unsqueeze(0) # (1, C, modes)
        
        Y_real = X_real * G_r - X_imag * G_i
        Y_imag = X_real * G_i + X_imag * G_r
        
        # 4. Transform back to time domain
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
        
        # Channel Mixing (MLP) - GELU is safe here because it's standard feature space
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
        
        # Using seq_length//2 + 1 modes to match standard FFT parameter count, but at custom resolution
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
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
                
        for block in self.mixers:
            for module in block.ffn:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(module.bias)
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, **kwargs):
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
# INFERENCE PIPELINE
# =============================================================================

def evaluate_mask(text, model, tokenizer, device, suppress_output=False):
    if "[MASK]" not in text:
        if not suppress_output: print(">> Error: Prompt must contain '[MASK]'.")
        return []

    # 1. Tokenize WITHOUT the [CLS] and [SEP] tokens the model never saw in training
    raw_inputs = tokenizer(text, add_special_tokens=False)["input_ids"]
    
    if len(raw_inputs) > 64:
        print(">> Error: Prompt is too long.")
        return []

    # 2. Create a generic Wikipedia-style buffer to give the waveform "energy"
    buffer_text = "The following is a historical and scientific text from an encyclopedia. " * 5
    buffer_ids = tokenizer(buffer_text, add_special_tokens=False)["input_ids"]

    # 3. Combine buffer + prompt to equal exactly 64 tokens (No zeros allowed)
    pad_length = 64 - len(raw_inputs)
    padded_ids = buffer_ids[:pad_length] + raw_inputs

    # Convert to tensor shape (1, 64)
    inputs = torch.tensor([padded_ids]).to(device)
    
    # Find where the mask token is in our new 64-token array
    mask_indices = torch.where(inputs == tokenizer.mask_token_id)
    if len(mask_indices[1]) == 0:
        return []
        
    mask_token_index = mask_indices[1][0] 
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=inputs)
        logits = outputs.logits
        
    mask_token_logits = logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=0).indices.tolist()
    
    results = []
    if not suppress_output: print("-" * 60)
    for i, token in enumerate(top_5_tokens):
        word = tokenizer.decode([token])
        filled_sentence = text.replace("[MASK]", f"**{word.strip()}**")
        results.append(word.strip())
        if not suppress_output:
            print(f"Rank {i+1}: {filled_sentence}")
    if not suppress_output: print("-" * 60)
    
    return results

# =============================================================================
# DATA PIPELINE
# =============================================================================

def compute_metrics(eval_pred):
    """Calculates accuracy metric during training for live monitoring."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    accuracy = (predictions[mask] == labels[mask]).mean()
    return {"accuracy": accuracy}

def get_datasets(dataset_name="wikitext-2-raw-v1", seq_length=64):
    print(f"Loading {dataset_name} from HuggingFace...")
    raw_datasets = load_dataset("Salesforce/wikitext", dataset_name)
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
    
    return lm_datasets["train"], lm_datasets["validation"].select(range(300)), data_collator, tokenizer

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext-2-raw-v1", choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=2048) 
    parser.add_argument("--num_modes", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=7) 
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--seq_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    # Parse known args so Colab's internal Jupyter args don't crash it
    args, _ = parser.parse_known_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    train_dataset, val_dataset, data_collator, tokenizer = get_datasets(args.dataset, args.seq_length)
    vocab_size = len(tokenizer)
    
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
        output_dir=DRIVE_PATH,
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=50,
        optim="adamw_torch_xla", # CRITICAL FIX: Safe optimizer for TPU XLA compilation
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        report_to="none" # Disable wandb for Colab execution
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics, # CRITICAL FIX: Track Accuracy!
    )
    
    trainer.train()
    
    # Save the final best weights to Drive
    final_path = os.path.join(DRIVE_PATH, "best_fractional_spectral.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Best weights saved to {final_path}")
    
    # ==========================================
    # POST-TRAINING DENSE CONTEXT INFERENCE TESTS
    # ==========================================
    print("\n" + "="*60)
    print("RUNNING DENSE CONTEXT INFERENCE TESTS")
    print("="*60)
    
    # Ensure model is on the correct device for inference
    device = xm.xla_device() if TPU_AVAILABLE else model.device
    model.to(device)
    
    test_prompts = [
        "Fourier transforms are extremely useful for processing [MASK] signals.",
        "The quick brown [MASK] jumps over the lazy dog.",
        "He went to the [MASK] to buy some groceries for dinner."
    ]
    
    for prompt in test_prompts:
        evaluate_mask(prompt, model, tokenizer, device)


if __name__ == "__main__":
    main()
