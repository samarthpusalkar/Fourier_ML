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
OUTPUT_PATH = "./TransformerBaseline_Checkpoints_BiggerDataset_GPU"
os.makedirs(OUTPUT_PATH, exist_ok=True)

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.05):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj.is_residual = True
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        
        # Fast, exact scaled dot-product attention (FlashAttention if available)
        y = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            dropout_p=self.attn_dropout.p if self.training else 0
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.05):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim)
        self.c_proj.is_residual = True
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(self.c_fc, self.gelu, self.c_proj, self.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class StandardTransformerLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, num_attention_heads=12, max_position_embeddings=512, dropout=0.05):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_attention_heads, dropout) 
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight # Exact weight tying
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'is_residual'):
                std *= (2 * len(self.blocks)) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, labels=None, **kwargs):
        device = input_ids.device
        B, T = input_ids.size()
        
        # Absolute positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        
        tok_emb = self.token_embedding(input_ids) # shape (b, t, n_embd)
        pos_emb = self.position_embedding(pos) # shape (1, t, n_embd)
        
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
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

def print_param_count(model):
    # Only count parameters that require gradients, and avoid double-counting tied weights
    num_params = len(set(p.data_ptr() for p in model.parameters() if p.requires_grad))
    actual_params = sum(p.numel() for p in set(p for p in model.parameters() if p.requires_grad))
    print(f"Total Trainable Parameters: {actual_params / 1e6:.2f}M")

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
    model = StandardTransformerLM(vocab_size=len(tokenizer), hidden_size=768, num_layers=12, num_attention_heads=12, max_position_embeddings=512)
    print("Initialized Standard Transformer Baseline (~110M Params)...")
    print_param_count(model)
    
    if flags.load_weights and os.path.exists(flags.load_weights):
        print(f"Loading weights from {flags.load_weights} (Streaming continuation mode)...")
        if flags.load_weights.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(flags.load_weights, device=device)
        else:
            state_dict = torch.load(flags.load_weights, map_location=device, weights_only=False)
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
        dataloader_num_workers=6, # not max to avoid request block
        dataloader_prefetch_factor=6, # shoould help avoind gpu to have free time
        dataloader_pin_memory=True # should help in memory bandwith transfer
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
