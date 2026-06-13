import os
import math
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.modeling_outputs import MaskedLMOutput

OUTPUT_PATH = "./FastFourierLM_Checkpoints"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# =============================================================================
# FAST FOURIER SPECTRAL ARCHITECTURE (Native cuFFT Accelerated)
# =============================================================================

class FastFourierMixer1D(nn.Module):
    def __init__(self, channels, seq_length):
        super().__init__()
        self.channels = channels
        self.seq_length = seq_length
        
        # When dealing with real numbers, the frequency domain is symmetric.
        # rfft cuts the sequence in half (seq_length // 2 + 1), saving 50% memory.
        self.freq_length = (seq_length // 2) + 1
        
        # 1. The Projections (creating v1 and v2)
        # We project the input into two separate spaces
        self.proj_v1 = nn.Linear(channels, channels)
        self.proj_v2 = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        
        # 2. The Parameter Barrier 'k' (Initialized in the frequency domain)
        # It must be complex because frequencies have magnitude and phase.
        self.k_freq = nn.Parameter(torch.randn(1, self.freq_length, channels, dtype=torch.cfloat) * 0.02)
        
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        # x shape: (batch_size, seq_length, channels)
        
        # Step 1: Generate v1 and v2
        v1 = self.proj_v1(x)
        v2 = self.activation(self.proj_v2(x))
        
        # Step 2: Move v1 to the frequency domain using Real FFT
        # Shape becomes (batch_size, freq_length, channels)
        v1_freq = torch.fft.rfft(v1, dim=1)
        
        # Step 3: Apply the Parameter Barrier 'k'
        # This is the exact equivalent of O(N^2) circulant matrix multiplication
        mixed_freq = v1_freq * self.k_freq
        
        # Step 4: Move back to the time domain
        v1_mixed = torch.fft.irfft(mixed_freq, n=self.seq_length, dim=1)
        
        # Step 5: Element-wise interaction with v2
        v3 = v1_mixed * v2
        
        # Final projection and residual connection
        return self.norm(self.out_proj(v3)) + x

class SpectralBlock(nn.Module):
    def __init__(self, latent_dim, seq_length):
        super().__init__()
        self.mixer = FastFourierMixer1D(latent_dim, seq_length)
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

class LargeSpectralLM(nn.Module):
    def __init__(self, vocab_size, seq_length=512, latent_dim=512, num_layers=15):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, latent_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(seq_length, latent_dim)
        self.mixers = nn.ModuleList([
            SpectralBlock(latent_dim, seq_length) 
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(latent_dim)
        
    def forward(self, input_ids, labels=None, **kwargs):
        z = self.embedding(input_ids) + self.pos_embedding(torch.arange(input_ids.shape[1], device=input_ids.device))
        for mixer in self.mixers: 
            z = mixer(z)
        logits = F.linear(self.ln_f(z), self.embedding.weight)
        loss = F.cross_entropy(logits.view(-1, self.embedding.weight.size(0)), labels.view(-1), ignore_index=-100) if labels is not None else None
        return MaskedLMOutput(loss=loss, logits=logits)

# =============================================================================
# DATA PIPELINE (Wikitext-103 Production Scale Setup)
# =============================================================================

def get_datasets(seq_length=512):
    print("Loading Wikitext-103 dataset from HuggingFace...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    tokenized = ds.map(lambda x: tokenizer(x["text"]), batched=True, remove_columns=["text"], num_proc=4)
    
    def group_texts(examples):
        concatenated_examples = {k: list(np.concatenate(examples[k])) for k in examples.keys() if len(examples[k]) > 0}
        if not concatenated_examples:
            return {k: [] for k in examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        total_length = (total_length // seq_length) * seq_length
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    print("Grouping tokens into sequences...")
    grouped = tokenized.map(group_texts, batched=True, num_proc=4)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    
    # Subsample validation to exactly 1000 items to guarantee speed and 0% chance of evaluation OOM
    val_dataset = grouped["validation"]
    if len(val_dataset) > 1000:
        val_dataset = val_dataset.select(range(1000))
        
    return grouped["train"], val_dataset, collator, tokenizer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple): 
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    return {"accuracy": (predictions[mask] == labels[mask]).mean()}

# =============================================================================
# MAIN RUNNER
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--load_weights", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device target: {device.upper()}")
    
    train_ds, val_ds, collator, tokenizer = get_datasets()
    model = LargeSpectralLM(len(tokenizer))
    
    if args.load_weights: 
        print(f"Loading weights from {args.load_weights}")
        model.load_state_dict(torch.load(args.load_weights, map_location=device))
        
    model.to(device)
    
    # Force optimize performance for Ada Lovelace Architecture (L40S)
    use_bf16 = torch.cuda.is_bf16_supported()
    if use_bf16:
        torch.backends.cuda.matmul.allow_tf32 = True 
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        num_train_epochs=4,              # Targeting your 2 epoch limit (~1 hour target window)
        per_device_train_batch_size=64, # Aggressive batching to maximize L40S 48GB VRAM
        per_device_eval_batch_size=4,   # Kept low to keep VRAM footprint negligible
        gradient_accumulation_steps=16,
        optim="adamw_torch",
        learning_rate=8.5e-05,
        lr_scheduler_type="cosine",      # Starts a brand new decay curve
        warmup_ratio = 0.05,
        # Circular Memory Protection for Saves
        save_strategy="steps",
        save_steps=100,                  
        save_total_limit=3,              # Keeps only the last 3 checkpoint files. Deletes older ones.
        
        # Anti-OOM Metric Gathering Management
        eval_strategy="steps",   
        eval_steps=100,                  
        eval_accumulation_steps=10,      # Offloads logit matrices to host CPU memory regularly
        
        logging_steps=10, 
        report_to="none",
        
        bf16=use_bf16,                   
        tf32=True,                       
        dataloader_num_workers=4         # Parallelizes data load to match high GPU processing throughput
    )
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_ds, 
        eval_dataset=val_ds, 
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from checkpoint: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        print("Starting clean training run on Wikitext-103...")
        trainer.train()
        
    final_path = os.path.join(OUTPUT_PATH, "best_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training finalized. Execution successful. Model saved to {final_path}")

if __name__ == "__main__":
    main()
