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

OUTPUT_PATH = "./HybridCausalFourierLM_Checkpoints"
os.makedirs(OUTPUT_PATH, exist_ok=True)

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

# =============================================================================
# HYBRID CONTINUOUS FOURIER ARCHITECTURE
# =============================================================================

class LinearFourierMixer(nn.Module):
    def __init__(self, channels, num_modes=128, num_heads=12):
        super().__init__()
        self.channels = channels
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        fractional = torch.exp(torch.linspace(math.log(0.0001), 0, num_modes))
        integers = torch.arange(1, num_modes + 1, dtype=torch.float32)
        freq_bands = torch.cat([fractional, integers])
        
        self.num_modes = freq_bands.shape[0]
        self.register_buffer("frequencies", freq_bands)

        self.q_proj = nn.Linear(channels, self.num_heads * self.num_modes)
        self.k_proj = nn.Linear(channels, self.num_heads * self.num_modes)
        self.v_proj = nn.Linear(channels, channels)
        
        self.proj_v2 = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, seq_len, C = x.shape
        
        # 1. Generate data-dependent Q and K
        # Use ELU + 1 to keep values positive for linear attention stability
        Q = F.elu(self.q_proj(x)).view(B, seq_len, self.num_heads, self.num_modes) + 1.0
        K = F.elu(self.k_proj(x)).view(B, seq_len, self.num_heads, self.num_modes) + 1.0
        
        v1 = self.v_proj(x)
        v2 = self.activation(self.proj_v2(x))
        
        # 2. Continuous time grid
        t = (torch.arange(seq_len, device=x.device, dtype=x.dtype) / 32).view(-1, 1)
        omega_t = 2 * math.pi * t * self.frequencies.unsqueeze(0) # (seq_len, num_modes)
       
        U = torch.cos(omega_t).unsqueeze(0).unsqueeze(2) # (1, seq_len, 1, num_modes)
        V = torch.sin(omega_t).unsqueeze(0).unsqueeze(2) # (1, seq_len, 1, num_modes)
        
        # 3. Modulate Q and K
        Q_cos = Q * U  # (B, seq_len, num_heads, num_modes)
        Q_sin = Q * V
        K_cos = K * U
        K_sin = K * V
        
        # 4. Compute Linear Mixing Matrix (A) efficiently without Softmax
        # A_ij = Q_cos_i * K_cos_j + Q_sin_i * K_sin_j
        A = torch.einsum('b i h m, b j h m -> b h i j', Q_cos, K_cos) + \
            torch.einsum('b i h m, b j h m -> b h i j', Q_sin, K_sin)
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype))
        A = A * causal_mask.unsqueeze(0).unsqueeze(0)
        
        # 5. Route Values
        v1_heads = v1.view(B, seq_len, self.num_heads, self.head_dim)
        
        # v1_token_mixed = A @ v1_heads
        v1_token_mixed = torch.einsum('b h i j, b j h d -> b i h d', A, v1_heads)
        
        v1_token_mixed = v1_token_mixed.reshape(B, seq_len, C)
        v3 = torch.addcmul(v1, v1_token_mixed, v2, value=1.0 / math.sqrt(seq_len))
        
        return self.norm(self.out_proj(v3)) + x

class SoftmaxFourierMixer(nn.Module):
    def __init__(self, channels, num_modes=128, num_heads=12):
        super().__init__()
        self.channels = channels
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        fractional = torch.exp(torch.linspace(math.log(0.0001), 0, num_modes))
        integers = torch.arange(1, num_modes + 1, dtype=torch.float32)
        freq_bands = torch.cat([fractional, integers])
        
        self.num_modes = freq_bands.shape[0]
        self.register_buffer("frequencies", freq_bands)

        self.q_proj = nn.Linear(channels, self.num_heads * self.num_modes)
        self.k_proj = nn.Linear(channels, self.num_heads * self.num_modes)
        self.v_proj = nn.Linear(channels, channels)
        
        self.proj_v2 = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, seq_len, C = x.shape
        
        Q = self.q_proj(x).view(B, seq_len, self.num_heads, self.num_modes)
        K = self.k_proj(x).view(B, seq_len, self.num_heads, self.num_modes)
        
        v1 = self.v_proj(x)
        v2 = self.activation(self.proj_v2(x))
        
        t = (torch.arange(seq_len, device=x.device, dtype=x.dtype) / 32).view(-1, 1)
        omega_t = 2 * math.pi * t * self.frequencies.unsqueeze(0) 
       
        U = torch.cos(omega_t).unsqueeze(0).unsqueeze(2) 
        V = torch.sin(omega_t).unsqueeze(0).unsqueeze(2) 
        
        Q_cos = Q * U  
        Q_sin = Q * V
        K_cos = K * U
        K_sin = K * V
        
        scale = 1.0 / math.sqrt(self.num_modes)
        
        A = torch.einsum('b i h m, b j h m -> b h i j', Q_cos, K_cos) * scale + \
            torch.einsum('b i h m, b j h m -> b h i j', Q_sin, K_sin) * scale
        
        # Causal Mask for Softmax (replace 0 with -inf)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        A = A.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        A = F.softmax(A, dim=-1)
        
        v1_heads = v1.view(B, seq_len, self.num_heads, self.head_dim)
        v1_token_mixed = torch.einsum('b h i j, b j h d -> b i h d', A, v1_heads)
        
        v1_token_mixed = v1_token_mixed.reshape(B, seq_len, C)
        v3 = torch.addcmul(v1, v1_token_mixed, v2, value=1.0 / math.sqrt(seq_len))
        
        return self.norm(self.out_proj(v3)) + x

class HybridSpectralBlock(nn.Module):
    def __init__(self, latent_dim, num_modes=128, is_softmax=False):
        super().__init__()
        if is_softmax:
            self.mixer = SoftmaxFourierMixer(latent_dim, num_modes)
        else:
            self.mixer = LinearFourierMixer(latent_dim, num_modes)
            
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

class HybridFourierLM(nn.Module):
    def __init__(self, vocab_size, latent_dim=768, num_layers=12, num_modes=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, latent_dim, padding_idx=0)
        
        # Hybrid Setup: [Linear, Linear, Linear, Softmax] repeating
        blocks = []
        for i in range(num_layers):
            is_softmax = (i % 4 == 3) # Layers 3, 7, 11 (0-indexed) will be Softmax
            blocks.append(HybridSpectralBlock(latent_dim, num_modes, is_softmax=is_softmax))
            
        self.mixers = nn.ModuleList(blocks)
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
        if "lm_head.weight" in sd:
            sd["lm_head.weight"] = sd["lm_head.weight"].clone()
        return sd

# =============================================================================
# DATA PIPELINE
# =============================================================================

def get_datasets(seq_length=512):
    print("Streaming FineWeb-Edu 10BT dataset from HuggingFace...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", model_max_length=int(1e9))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
                    if worker_info is not None:
                        iterator = itertools.islice(iterator, worker_info.id, None, worker_info.num_workers)
                        
                    for item in iterator:
                        yield item
                        items_yielded += 1
                except Exception as e:
                    logging.warning(f"Stream interrupted due to error: {e}. Restarting stream...")
                
                if items_yielded == 0:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    flags, _ = parser.parse_known_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Process initialized on device: {device.upper()}")
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_ds, val_ds, collator, tokenizer = get_datasets(seq_length=flags.seq_len)
    model = HybridFourierLM(vocab_size=len(tokenizer), latent_dim=768, num_layers=12, num_modes=128)
    
    if flags.load_weights and os.path.exists(flags.load_weights):
        state_dict = torch.load(flags.load_weights, map_location=device)
        model.load_state_dict(state_dict)
        
    model.to(device)
    
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        max_steps=flags.max_steps,
        per_device_train_batch_size=flags.batch_size, 
        per_device_eval_batch_size=flags.batch_size,   
        gradient_accumulation_steps=1,
        
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",       
        torch_compile=True if torch.cuda.is_available() else False,              
        bf16=use_bf16,                   
        tf32=True if torch.cuda.is_available() else False,                       
        
        learning_rate=flags.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",      
        warmup_ratio=0.05,
        
        save_strategy="steps",
        save_steps=2000,   
        save_total_limit=2,
        
        push_to_hub=False, 
        report_to="none",
        
        dataloader_num_workers=14 if torch.cuda.is_available() else 0,
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
    
    print("Starting hybrid massive scale causal training...")
    trainer.train(resume_from_checkpoint=flags.resume_from_checkpoint)
        
    final_path = os.path.join(OUTPUT_PATH, "best_model_streaming")
    trainer.save_model(final_path)

if __name__ == "__main__":
    main()
