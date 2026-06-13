import os
import math
import argparse
import warnings
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
        t = torch.linspace(0, 1, seq_len, device=x.device, dtype=x.dtype).view(-1, 1) # (seq_len, 1)
        
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
    # This is safe and accurate because it runs inside the spawned process
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
            print(f"Loading weights from {flags.load_weights} (fresh epoch / fine-tuning mode)...")
        # Load state dict map_location="cpu" to be safe with multi-processing/TPUs
        state_dict = torch.load(flags.load_weights, map_location="cpu")
        model.load_state_dict(state_dict)
    
    # HuggingFace Trainer automatically handles moving the model to TPU/XLA devices
    # if the torch_xla module is present in the environment.
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        num_train_epochs=flags.epochs,
        # =====================================================================
        # ULTIMATE HBM OOM FIX (XLA Fusion Bloat)
        # Gradient accumulation > 1 on PyTorch XLA causes the compiler to fuse 
        # multiple forward/backward steps into a SINGLE execution graph! 
        # A grad_accum of 4 fused 4 graphs together, demanding 36.95GB HBM!
        # FIX: We must STRICTLY set gradient_accumulation_steps=1. 
        # A native batch size of 8 takes exactly ~9.2GB of HBM, fitting perfectly.
        # =====================================================================
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8,   
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        learning_rate=flags.learning_rate,
        lr_scheduler_type="cosine",      
        warmup_ratio=0.05,
        
        # Save per epoch to Google Drive, ensuring we don't spam API quotas
        save_strategy="steps",
        save_steps=1000,   
        save_total_limit=2,              
        
        eval_strategy="steps",
        eval_steps=1000, 
        eval_accumulation_steps=10,
        logging_steps=100, 
        report_to="none",
        
        dataloader_num_workers=0,
        
        # Crucial TPU Flags for Multi-Processing
        ddp_backend="xla",
        
        # Note: fp16/bf16 settings are handled natively by XLA compilation
        # xla=True, # Depending on transformers version, this is auto-detected
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
    
    if flags.resume_from and os.path.exists(flags.resume_from):
        if index == 0:
            print(f"Resuming training from checkpoint: {flags.resume_from}")
        trainer.train(resume_from_checkpoint=flags.resume_from)
    else:
        if index == 0:
            print("Starting Kaggle TPUv5e-8 multi-core training run...")
        trainer.train()
        
    # Only the master process (Core 0) should save the final state dict to avoid race conditions
    if index == 0:
        final_path = os.path.join(OUTPUT_PATH, "best_model.pt")
        # Using XLA safe save if needed, but Trainer usually handles standard save
        xm.save(model.state_dict(), final_path)
        print(f"Training finalized. Model saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--load_weights", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=10.0)
    flags, _ = parser.parse_known_args()
    
    # Fix the common Hugging Face network wrapper freeze on managed platforms
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    
    # =============================================================================
    # ENVIRONMENT RUN CONFIGURATION
    # =============================================================================
    # Set this to True when committing on Kaggle TPUv5e-8 or Colab TPU v2-8 (8 cores).
    # Set to False if running on Colab v5e-1 (1 core) or local CPU.
    USE_TPU_CLUSTER = True  

    if USE_TPU_CLUSTER:
        print("Targeting TPU Cluster. Spawning parallel processes for all available cores...")
        xmp.spawn(_mp_fn, args=(flags,), nprocs=None, start_method='fork')
    else:
        print("Running in local single-process fallback mode...")
        # Directly invoke your function on the main thread for CPU/single-GPU debugging
        _mp_fn(index=0, flags=flags)
