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
from transformers.modeling_outputs import CausalLMOutput

OUTPUT_PATH = "./CausalFourierLM_Checkpoints"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# =============================================================================
# CAUSAL CONTINUOUS FOURIER ARCHITECTURE (~120M Parameter Scale)
# =============================================================================

class CausalContinuousFourierMixer1D(nn.Module):
    def __init__(self, channels, num_modes=128):
        super().__init__()
        self.channels = channels
        self.num_modes = num_modes
        
        # ---------------------------------------------------------------------
        # THE INFINITE DIMENSIONAL WEIGHTS (Fourier Coefficient Training)
        # ---------------------------------------------------------------------
        self.fourier_amplitudes = nn.Parameter(torch.randn(channels, num_modes) / math.sqrt(num_modes))
        self.fourier_phases = nn.Parameter(torch.randn(channels, num_modes))
        
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
        
        # =====================================================================
        # PART 1: TOKEN MIXING (Causal Zero-Padding Theorem)
        # =====================================================================
        # Create a continuous time grid normalized from 0 to 1
        t = torch.linspace(0, 1, seq_len, device=x.device).view(-1, 1) # (seq_len, 1)
        
        args = 2 * math.pi * t * self.frequencies.unsqueeze(0) # (seq_len, num_modes)
        args = args.unsqueeze(-1) + self.fourier_phases.T.unsqueeze(0) # (seq_len, num_modes, C)
        
        # Render the EXACT time-domain barrier 'k' for the current dynamic length
        k_time = (torch.cos(args) * self.fourier_amplitudes.T.unsqueeze(0)).sum(dim=1) # (seq_len, C)
        
        # STRICT CAUSALITY ENFORCEMENT
        # To make circular convolution linear (and causal), we pad both signals to 2 * seq_len
        pad_len = seq_len
        v1_padded = F.pad(v1, (0, 0, 0, pad_len)) # Shape: (B, 2*seq_len, C)
        k_time_padded = F.pad(k_time, (0, 0, 0, pad_len)) # Shape: (2*seq_len, C)
        
        # FFT over the padded sequence length
        v1_seq_freq = torch.fft.rfft(v1_padded, dim=1)
        k_seq_freq = torch.fft.rfft(k_time_padded, dim=0).unsqueeze(0)
        
        # Multiply and inverse FFT
        v1_token_mixed_padded = torch.fft.irfft(v1_seq_freq * k_seq_freq, n=2*seq_len, dim=1)
        
        # Slice off the padding. This guarantees mathematically perfect causality.
        v1_token_mixed = v1_token_mixed_padded[:, :seq_len, :]
        
        # =====================================================================
        # PART 2: VECTOR / EMBEDDING MIXING
        # =====================================================================
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
        
        # Since we want to support infinite context lengths without cap, 
        # a standard absolute positional embedding with a max_length cap is a bottleneck.
        # However, because the Continuous Fourier Mixer already incorporates time 't' 
        # dynamically as a continuous grid across the sequence, explicit positional 
        # embeddings are mathematically redundant! The layer is fundamentally position-aware.
        
        self.mixers = nn.ModuleList([
            CausalSpectralBlock(latent_dim, num_modes) 
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(latent_dim)
        
        # Weight tying for head projection
        self.lm_head = nn.Linear(latent_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
    def forward(self, input_ids, labels=None, **kwargs):
        z = self.embedding(input_ids)
        
        for mixer in self.mixers: 
            z = mixer(z)
            
        logits = self.lm_head(self.ln_f(z))
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return CausalLMOutput(loss=loss, logits=logits)

# =============================================================================
# DATA PIPELINE (Wikitext-103 Causal Language Modeling)
# =============================================================================

def get_datasets(seq_length=512):
    print("Loading Wikitext-103 dataset from HuggingFace...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # BERT tokenizer usually has max_length caps, but we just use it for encoding words.
    tokenized = ds.map(lambda x: tokenizer(x["text"], add_special_tokens=False), batched=True, remove_columns=["text"], num_proc=4)
    
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
        # For Causal LM, labels are exactly the input_ids (shifted inside the model)
        result["labels"] = result["input_ids"].copy()
        return result

    print("Grouping tokens into sequences...")
    grouped = tokenized.map(group_texts, batched=True, num_proc=4)
    
    # mlm=False tells the collator NOT to mask tokens. It just batches them.
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    val_dataset = grouped["validation"]
    if len(val_dataset) > 1000:
        val_dataset = val_dataset.select(range(1000))
        
    return grouped["train"], val_dataset, collator, tokenizer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple): 
        logits = logits[0]
        
    # Account for the shift in predictions and labels
    predictions = np.argmax(logits[..., :-1, :], axis=-1)
    shifted_labels = labels[..., 1:]
    
    mask = shifted_labels != -100
    accuracy = (predictions[mask] == shifted_labels[mask]).mean()
    return {"accuracy": accuracy}

# =============================================================================
# MAIN RUNNER
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--load_weights", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    args, _ = parser.parse_known_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device target: {device.upper()}")
    
    train_ds, val_ds, collator, tokenizer = get_datasets(seq_length=args.seq_len)
    
    # GPT-1 Scale (110M) configuration
    # latent_dim=768, num_layers=12, vocab_size=~30k
    model = ContinuousFourierLM(vocab_size=len(tokenizer), latent_dim=768, num_layers=12, num_modes=128)
    
    # Print Parameter Count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {total_params / 1e6:.2f}M Parameters")
    
    if args.load_weights: 
        print(f"Loading weights from {args.load_weights}")
        model.load_state_dict(torch.load(args.load_weights, map_location=device))
        
    model.to(device)
    
    # Force optimize performance
    use_bf16 = torch.cuda.is_bf16_supported()
    if use_bf16:
        torch.backends.cuda.matmul.allow_tf32 = True 
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        num_train_epochs=2,              
        per_device_train_batch_size=32,  # Batch size adjusted for 768 dim
        per_device_eval_batch_size=4,   
        gradient_accumulation_steps=8,
        optim="adamw_torch",
        learning_rate=1.5e-04,           # Slightly higher LR for 100M parameter models
        lr_scheduler_type="cosine",      
        warmup_ratio=0.05,
        save_strategy="steps",
        save_steps=500,                  
        save_total_limit=3,              
        eval_strategy="steps",   
        eval_steps=500,                  
        eval_accumulation_steps=10,      
        logging_steps=50, 
        report_to="none",
        bf16=use_bf16,                   
        tf32=True,                       
        dataloader_num_workers=4         
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
        print("Starting causal autoregressive training run on Wikitext-103...")
        trainer.train()
        
    final_path = os.path.join(OUTPUT_PATH, "best_model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training finalized. Model saved to {final_path}")

if __name__ == "__main__":
    main()
