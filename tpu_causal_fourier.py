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

# =============================================================================
# GOOGLE COLAB & TPU SETUP
# =============================================================================
try:
    from google.colab import drive
    print("Detected Google Colab environment. Mounting Google Drive...")
    drive.mount('/content/drive')
    OUTPUT_PATH = "/content/drive/MyDrive/CausalFourierLM_Checkpoints"
except ImportError:
    print("Not running in Colab. Using local checkpoint directory.")
    OUTPUT_PATH = "./CausalFourierLM_Checkpoints"

os.makedirs(OUTPUT_PATH, exist_ok=True)

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

# =============================================================================
# CAUSAL CONTINUOUS FOURIER ARCHITECTURE (~120M Parameter Scale)
# =============================================================================

class CausalContinuousFourierMixer1D(nn.Module):
    def __init__(self, channels, num_modes=128):
        super().__init__()
        self.channels = channels
        self.num_modes = num_modes
        
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
        
        # Create continuous time grid
        t = torch.linspace(0, 1, seq_len, device=x.device).view(-1, 1) # (seq_len, 1)
        
        args = 2 * math.pi * t * self.frequencies.unsqueeze(0) # (seq_len, num_modes)
        args = args.unsqueeze(-1) + self.fourier_phases.T.unsqueeze(0) # (seq_len, num_modes, C)
        
        k_time = (torch.cos(args) * self.fourier_amplitudes.T.unsqueeze(0)).sum(dim=1) # (seq_len, C)
        
        # Causal padding
        pad_len = seq_len
        v1_padded = F.pad(v1, (0, 0, 0, pad_len)) 
        k_time_padded = F.pad(k_time, (0, 0, 0, pad_len)) 
        
        # FFT token mixing
        v1_seq_freq = torch.fft.rfft(v1_padded, dim=1)
        k_seq_freq = torch.fft.rfft(k_time_padded, dim=0).unsqueeze(0)
        
        v1_token_mixed_padded = torch.fft.irfft(v1_seq_freq * k_seq_freq, n=2*seq_len, dim=1)
        v1_token_mixed = v1_token_mixed_padded[:, :seq_len, :]
        
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

# =============================================================================
# DATA PIPELINE
# =============================================================================

def get_datasets(seq_length=512):
    print("Loading Wikitext-103 dataset from HuggingFace...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    tokenized = ds.map(lambda x: tokenizer(x["text"], add_special_tokens=False), batched=True, remove_columns=["text"], num_proc=4)
    
    def group_texts(examples):
        concatenated_examples = {k: list(np.concatenate(examples[k])) for k in examples.keys() if len(examples[k]) > 0}
        if not concatenated_examples:
            return {k: [] for k in examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        # Extremely important for TPU: guarantee fixed size to avoid XLA recompilation
        total_length = (total_length // seq_length) * seq_length
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print("Grouping tokens into sequences...")
    grouped = tokenized.map(group_texts, batched=True, num_proc=4)
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    val_dataset = grouped["validation"].select(range(1000)) if len(grouped["validation"]) > 1000 else grouped["validation"]
        
    return grouped["train"], val_dataset, collator, tokenizer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple): 
        logits = logits[0]
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
    parser.add_argument("--seq_len", type=int, default=512)
    args, _ = parser.parse_known_args()
    
    train_ds, val_ds, collator, tokenizer = get_datasets(seq_length=args.seq_len)
    model = ContinuousFourierLM(vocab_size=len(tokenizer), latent_dim=768, num_layers=12, num_modes=128)
    
    # HuggingFace Trainer automatically handles moving the model to TPU/XLA devices
    # if the torch_xla module is present in the environment.
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        num_train_epochs=3,
        # TPUs have massive HBM per core, higher batch sizes are critical for speed
        per_device_train_batch_size=64, 
        per_device_eval_batch_size=16,   
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        learning_rate=2e-04,
        lr_scheduler_type="cosine",      
        warmup_ratio=0.05,
        
        # Save per epoch to Google Drive, ensuring we don't spam API quotas
        save_strategy="epoch",
        save_total_limit=2,              
        
        eval_strategy="epoch",   
        logging_steps=100, 
        report_to="none",
        
        dataloader_num_workers=4,
        
        # Note: fp16/bf16 settings are handled natively by XLA compilation
        # xla=True, # Depending on transformers version, this is auto-detected
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
        print("Starting Colab TPU causal training run...")
        trainer.train()
        
    final_path = os.path.join(OUTPUT_PATH, "best_model.pt")
    # Using XLA safe save if needed, but Trainer usually handles standard save
    torch.save(model.state_dict(), final_path)
    print(f"Training finalized. Model saved to {final_path}")

if __name__ == "__main__":
    main()
