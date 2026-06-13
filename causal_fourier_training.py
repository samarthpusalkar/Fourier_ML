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
    def __init__(self, channels, num_modes=128, num_heads=12):
        super().__init__()
        self.channels = channels
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # ---------------------------------------------------------------------
        # MULTI-HEAD CONTINUOUS FOURIER WEIGHTS
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
        # THE FINAL COMPILER FIX (Pure Matrix Multiplication)
        # We replace all 'einsum' calls with explicit, standard 'matmul' and 
        # '.contiguous()' reshaping. XLA sometimes assigns broken memory strides 
        # (tensor_data crash) when handling heavily interleaved einsum operations.
        # =====================================================================
        
        # 1. Generate Query Matrices (Q) by rotating the weights through time
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
        K_expanded = K_matrix.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        
        # MatMul: (1, H, seq_len, seq_len) @ (B, H, seq_len, D) -> (B, H, seq_len, D)
        v1_token_mixed = torch.matmul(K_expanded, v1_heads)
        
        # 5. Flatten back to standard format safely
        # Explicit contiguous() prevents memory stride bugs (tensor_data crash)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--load_weights", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--optim", type=str, choices=["adamw_torch", "adafactor"], default="adafactor",
                        help="Optimizer to use for training (adamw_torch or adafactor)")
    parser.add_argument("--enable_momentum", action="store_true", default=False,
                        help="Enable momentum (beta1=0.9) in Adafactor. Slightly increases VRAM.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Peak learning rate for the training run.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay coefficient.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-device training batch size.")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--epochs", type=float, default=2.0,
                        help="Number of training epochs.")
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
        if args.load_weights.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.load_weights)
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(torch.load(args.load_weights, map_location=device))
        
    model.to(device)
    
    # Force optimize performance
    use_bf16 = torch.cuda.is_bf16_supported()
    if use_bf16:
        torch.backends.cuda.matmul.allow_tf32 = True 
    
    # Configure custom optimizer if Adafactor is chosen
    optimizers = (None, None)
    if args.optim == "adafactor":
        from transformers import Adafactor
        print("Configuring custom Adafactor optimizer with relative_step=False...")
        
        # Exclude bias and norm parameters from weight decay
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "LayerNorm", "layernorm", "norm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=0.9 if args.enable_momentum else None,
            weight_decay=args.weight_decay,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False
        )
        optimizers = (optimizer, None)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        num_train_epochs=args.epochs,              
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,   
        gradient_accumulation_steps=args.grad_accum,
        optim=args.optim,
        learning_rate=args.learning_rate,
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
        tf32=torch.cuda.is_available(),  # Enable TF32 for Ampere/Ada Lovelace architecture
        save_safetensors=False,          # Disable safetensors saving due to weight tying (shared memory)
        dataloader_num_workers=4         
    )
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_ds, 
        eval_dataset=val_ds, 
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        optimizers=optimizers
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
