import os
import math
import argparse
import warnings
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Clear Kaggle TPU environment variables that conflict with PyTorch XLA PJRT
os.environ.pop('TPU_PROCESS_ADDRESSES', None)
os.environ.pop('CLOUD_TPU_TASK_ID', None)

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"

# =============================================================================
# GOOGLE COLAB & TPU SETUP
# =============================================================================
try:
    from google.colab import drive
    print("Detected Google Colab environment. Mounting Google Drive...")
    drive.mount('/content/drive')
    os.environ["HF_HOME"] = "/content/drive/MyDrive/HF_Cache"
    OUTPUT_PATH = "/content/drive/MyDrive/CausalFourierLM_Checkpoints_BiggerDataset"
except:
    print("Not running in Colab. Using local checkpoint directory.")
    OUTPUT_PATH = "./CausalFourierLM_Checkpoints_BiggerDataset"

os.makedirs(OUTPUT_PATH, exist_ok=True)

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

# =============================================================================
# CAUSAL CONTINUOUS FOURIER ARCHITECTURE (~120M Parameter Scale)
# =============================================================================

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
        
        # Log-spaced frequencies from 0.01 cycles to 128 cycles (Nyquist limit)
        # This provides both infinite context scaling (fractional frequencies) 
        # and sharp local attention (high frequencies) natively!
        freq_bands = torch.logspace(math.log10(0.01), math.log10(128.0), num_modes)
        self.register_buffer("frequencies", freq_bands)

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
        # FIX: We MUST use absolute scaling based on the base training length (512).
        # During training (seq_len=512), steps are exactly 1/511. 
        t = (torch.arange(seq_len, device=x.device, dtype=x.dtype) / 511.0).view(-1, 1)
        
        omega_t = 2 * math.pi * t * self.frequencies.unsqueeze(0) # (seq_len, num_modes)
        U = torch.cos(omega_t) # (seq_len, num_modes)
        V = torch.sin(omega_t) # (seq_len, num_modes)
        
        W_cos = self.fourier_amplitudes * torch.cos(self.fourier_phases) # (num_heads, num_modes)
        W_sin = self.fourier_amplitudes * torch.sin(self.fourier_phases) # (num_heads, num_modes)
        
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
        M_cos = torch.matmul(P_cos_h, U.T) # (num_heads, seq_len, seq_len)
        M_sin = torch.matmul(P_sin_h, V.T)
        K_matrix = M_cos + M_sin 
        
        # 3. Apply standard Causal Mask safely
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype))
        K_matrix = K_matrix * causal_mask.unsqueeze(0)
        
        # 4. Route Values (v1) using standard Attention Batched MatMul
        v1_heads = v1.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Explicitly expanding the batch dimension for XLA matmul safety
        K_expanded = K_matrix.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        
        v1_token_mixed = torch.matmul(K_expanded, v1_heads)
        
        # 5. Flatten back to standard format safely
        v1_token_mixed = v1_token_mixed.transpose(1, 2).contiguous().view(B, seq_len, C)
        
        # Scale to prevent activation variance explosion
        v1_token_mixed = v1_token_mixed / math.sqrt(seq_len)
        
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

# =============================================================================
# DATA PIPELINE (STREAMING OPENWEBTEXT / FINEWEB-EDU)
# =============================================================================

def get_datasets(seq_length=512):
    print("Streaming FineWeb-Edu 10BT dataset from HuggingFace...")
    # FineWeb-Edu is the highest quality next-token prediction dataset available today.
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_and_group(examples):
        tokenized = tokenizer(examples["text"], add_special_tokens=False)
        concatenated_ids = []
        for ids in tokenized["input_ids"]:
            concatenated_ids.extend(ids)
            
        total_length = len(concatenated_ids)
        # Drop the remainder that doesn't fit into seq_length cleanly
        total_length = (total_length // seq_length) * seq_length
        
        if total_length == 0:
            return {"input_ids": [], "labels": []}
            
        result_ids = [concatenated_ids[i : i + seq_length] for i in range(0, total_length, seq_length)]
        
        return {"input_ids": result_ids, "labels": result_ids.copy()}

    print("Mapping tokenizer to stream...")
    # Safe fallback if features are not loaded properly on iterable dataset
    columns_to_remove = ["text", "id", "dump", "url", "date", "file_path", "language", "language_score", "token_count", "score", "int_score"]
    
    streamed_dataset = ds.map(tokenize_and_group, batched=True, remove_columns=columns_to_remove)
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Take first 1000 items from the stream as a rolling validation set
    val_dataset = streamed_dataset.take(1000)
    
    # Skip the first 1000 to use the rest for training
    hf_train_dataset = streamed_dataset.skip(1000)
    
    # Wrap the dataset in an infinite generator to prevent Trainer from triggering
    # epoch-boundary shuffles (which crash due to .skip()/.take()) and to 
    # seamlessly recover from Hugging Face network stream timeouts.
    class InfiniteStreamer(torch.utils.data.IterableDataset):
        def __init__(self, dataset):
            self.dataset = dataset
        def __iter__(self):
            while True:
                items_yielded = 0
                try:
                    # Attempt to iterate through the dataset
                    for item in self.dataset:
                        yield item
                        items_yielded += 1
                except Exception as e:
                    # If the network drops mid-stream, catch the exception!
                    # We log it, and the `while True` loop will naturally restart the stream.
                    logging.warning(f"Stream interrupted due to error: {e}. Restarting stream...")
                
                # Safety net: prevent an infinite CPU hang if the network is permanently dead
                if items_yielded == 0:
                    print("Hugging Face stream returned 0 items (network dead). Stopping to prevent infinite hang.")
                    break

    train_dataset = InfiniteStreamer(hf_train_dataset)
    
    return train_dataset, val_dataset, collator, tokenizer

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits[..., :-1, :].argmax(dim=-1)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    shifted_labels = labels[..., 1:]
    mask = shifted_labels != -100
    accuracy = (predictions[mask] == shifted_labels[mask]).mean()
    return {"accuracy": accuracy}

# =============================================================================
# MAIN RUNNER
# =============================================================================

def _mp_fn(index, flags):
    device = xm.xla_device()
    try:
        import torch_xla.runtime as xr
        world_size = xr.world_size()
    except (ImportError, AttributeError):
        try:
            world_size = xm.xrt_world_size()
        except Exception:
            world_size = 1
    print(f"Process {index} / {world_size} initialized cleanly on device: {device}")
    
    train_ds, val_ds, collator, tokenizer = get_datasets(seq_length=flags.seq_len)
    model = ContinuousFourierLM(vocab_size=len(tokenizer), latent_dim=768, num_layers=12, num_modes=128)
    
    if flags.load_weights and os.path.exists(flags.load_weights):
        if index == 0:
            print(f"Loading weights from {flags.load_weights} (Streaming continuation mode)...")
        state_dict = torch.load(flags.load_weights, map_location="cpu")
        model.load_state_dict(state_dict)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        # We must use max_steps instead of num_train_epochs because IterableDatasets do not have a fixed length!
        max_steps=flags.max_steps,
        
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8,   
        gradient_accumulation_steps=1,
        optim="adafactor",
        optim_args="relative_step=False,scale_parameter=False,warmup_init=False",
        learning_rate=flags.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",      
        warmup_ratio=0.05,
        
        # Save checkpoints every 2000 steps to Google Drive (or Kaggle local disk)
        save_strategy="steps",
        save_steps=2000,   
        save_total_limit=2,
        
        # Cloud Checkpoint Backup
        push_to_hub=True,
        hub_strategy="checkpoint",         # CRITICAL: Uploads the actual crash-recovery checkpoint folders
        hub_private_repo=True,
        hub_model_id="your-username/my-crash-backups",              
        
        eval_strategy="steps",
        eval_steps=1000, 
        eval_accumulation_steps=10,
        logging_steps=100, 
        report_to="none",
        
        dataloader_num_workers=0,
        ddp_backend="xla",
    )
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_ds, 
        eval_dataset=val_ds, 
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    
    if index == 0:
        print("Starting Kaggle TPUv5e-8 Data Streaming run...")
    trainer.train(resume_from_checkpoint=flags.resume_from_checkpoint)
        
    if index == 0:
        final_path = os.path.join(OUTPUT_PATH, "best_model_streaming.pt")
        xm.save(model.state_dict(), final_path)
        print(f"Training finalized. Model saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--max_steps", type=int, default=50000)
    flags, _ = parser.parse_known_args()
    
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    
    USE_TPU_CLUSTER = True  

    if USE_TPU_CLUSTER:
        print("Targeting TPU Cluster. Spawning parallel processes for streaming...")
        xmp.spawn(_mp_fn, args=(flags,), nprocs=None, start_method='fork')
    else:
        print("Running in local single-process fallback mode...")
        _mp_fn(index=0, flags=flags)
