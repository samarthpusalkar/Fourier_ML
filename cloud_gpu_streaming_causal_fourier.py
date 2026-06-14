import os
import math
import argparse
import warnings
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
OUTPUT_PATH = "./CausalFourierLM_Checkpoints_BiggerDataset_GPU"
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
        
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        # Safetensors throws a fatal error if it detects shared memory (tied weights).
        # We clone the lm_head weight in the state_dict so it saves safely to disk.
        # When loaded, PyTorch correctly restores it into the tied memory block!
        if "lm_head.weight" in sd:
            sd["lm_head.weight"] = sd["lm_head.weight"].clone()
        return sd

# =============================================================================
# DATA PIPELINE (STREAMING OPENWEBTEXT / FINEWEB-EDU)
# =============================================================================

def get_datasets(seq_length=512):
    print("Streaming FineWeb-Edu 10BT dataset from HuggingFace...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    # Force the tokenizer to ignore document lengths so it stops warning about > 512 chunks
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=int(1e9))
    
    def tokenize_and_group(examples):
        tokenized = tokenizer(examples["text"], add_special_tokens=False)
        concatenated_ids = []
        for ids in tokenized["input_ids"]:
            concatenated_ids.extend(ids)
            
        total_length = len(concatenated_ids)
        total_length = (total_length // seq_length) * seq_length
        
        if total_length == 0:
            return {"input_ids": [], "labels": []}
            
        result_ids = [concatenated_ids[i : i + seq_length] for i in range(0, total_length, seq_length)]
        
        return {"input_ids": result_ids, "labels": result_ids.copy()}

    print("Mapping tokenizer to stream...")
    columns_to_remove = ["text", "id", "dump", "url", "date", "file_path", "language", "language_score", "token_count", "score", "int_score"]
    
    streamed_dataset = ds.map(tokenize_and_group, batched=True, remove_columns=columns_to_remove)
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    val_dataset = streamed_dataset.take(1000)
    hf_train_dataset = streamed_dataset.skip(1000)
    
    class InfiniteStreamer(torch.utils.data.IterableDataset):
        def __init__(self, dataset):
            self.dataset = dataset
        def __iter__(self):
            import itertools
            worker_info = torch.utils.data.get_worker_info()
            while True:
                items_yielded = 0
                try:
                    iterator = iter(self.dataset)
                    # Safely shard the dataset stream across PyTorch workers so they don't duplicate data
                    if worker_info is not None:
                        iterator = itertools.islice(iterator, worker_info.id, None, worker_info.num_workers)
                        
                    for item in iterator:
                        yield item
                        items_yielded += 1
                except Exception as e:
                    logging.warning(f"Stream interrupted due to error: {e}. Restarting stream...")
                
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
# MAIN RUNNER (H100 GPU OPTIMIZED)
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--max_steps", type=int, default=50000)
    # H100 Optimization Args
    # An H100 80GB has massive VRAM. Defaulting to 128 to saturate the Tensor Cores!
    parser.add_argument("--batch_size", type=int, default=128, help="H100 native batch size per device")
    flags, _ = parser.parse_known_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Process initialized cleanly on device: {device.upper()}")
    
    # H100 GPU Optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_ds, val_ds, collator, tokenizer = get_datasets(seq_length=flags.seq_len)
    model = ContinuousFourierLM(vocab_size=len(tokenizer), latent_dim=768, num_layers=12, num_modes=128)
    
    if flags.load_weights and os.path.exists(flags.load_weights):
        print(f"Loading weights from {flags.load_weights} (Streaming continuation mode)...")
        state_dict = torch.load(flags.load_weights, map_location=device)
        model.load_state_dict(state_dict)
        
    model.to(device)
    
    use_bf16 = torch.cuda.is_bf16_supported()
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        max_steps=flags.max_steps,
        
        # H100 optimized batch size
        per_device_train_batch_size=flags.batch_size, 
        per_device_eval_batch_size=flags.batch_size,   
        gradient_accumulation_steps=1,
        
        # H100 Premium Optimizations
        optim="adamw_torch_fused",       # Extremely fast natively fused optimizer for Ampere/Hopper
        torch_compile=True,              # JIT compiles the model graph for massive speedups
        bf16=use_bf16,                   # Native Brain Float 16
        tf32=True,                       # TensorFloat32 Core utilization
        
        learning_rate=flags.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",      
        warmup_ratio=0.05,
        
        save_strategy="steps",
        save_steps=2000,   
        save_total_limit=2,
        
        push_to_hub=True,
        hub_strategy="checkpoint",         
        hub_private_repo=True,
        hub_model_id="your-username/my-crash-backups",              
        
        eval_strategy="steps",
        eval_steps=1000, 
        eval_accumulation_steps=10,
        logging_steps=100, 
        report_to="none",
        
        # Now fully safe to use due to the InfiniteStreamer sharding fix!
        dataloader_num_workers=14,
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
    
    print("Starting massive scale causal training run with streaming dataset on GPU...")
    trainer.train(resume_from_checkpoint=flags.resume_from_checkpoint)
        
    # Let Hugging Face Trainer save the final model so it correctly strips 
    # the `_orig_mod.` prefixes generated by torch_compile!
    final_path = os.path.join(OUTPUT_PATH, "best_model_streaming")
    trainer.save_model(final_path)
    print(f"Training finalized. Model perfectly saved and stripped of compile prefixes to {final_path}")

if __name__ == "__main__":
    main()
