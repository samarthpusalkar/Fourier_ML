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
    OUTPUT_PATH = "/content/drive/MyDrive/TransformerBaseline_Checkpoints_BiggerDataset"
except:
    print("Not running in Colab. Using local checkpoint directory.")
    OUTPUT_PATH = "./TransformerBaseline_Checkpoints_BiggerDataset_TPU"

os.makedirs(OUTPUT_PATH, exist_ok=True)

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

# =============================================================================
# STANDARD TRANSFORMER BASELINE (GPT-2 ARCHITECTURE)
# =============================================================================

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
        
        # Fast, exact scaled dot-product attention
        # NOTE: TPU natively supports this operator beautifully via XLA lowering
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
# MAIN RUNNER (TPU OPTIMIZED)
# =============================================================================

def print_param_count(model):
    # Only count parameters that require gradients, and avoid double-counting tied weights
    num_params = len(set(p.data_ptr() for p in model.parameters() if p.requires_grad))
    actual_params = sum(p.numel() for p in set(p for p in model.parameters() if p.requires_grad))
    print(f"Total Trainable Parameters: {actual_params / 1e6:.2f}M")

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
    model = StandardTransformerLM(vocab_size=len(tokenizer), hidden_size=768, num_layers=12, num_attention_heads=12, max_position_embeddings=512)
    
    if index == 0:
        print("Initialized Standard Transformer Baseline (~110M Params)...")
        print_param_count(model)
        
    if flags.load_weights and os.path.exists(flags.load_weights):
        if index == 0:
            print(f"Loading weights from {flags.load_weights} (Streaming continuation mode)...")
        if flags.load_weights.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(flags.load_weights, device="cpu")
        else:
            state_dict = torch.load(flags.load_weights, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        max_steps=flags.max_steps,
        
        # Batch size 8 per TPU core matches original Fourier script
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8,   
        gradient_accumulation_steps=1,
        
        # TPU Optimizations
        optim="adafactor",
        optim_args="relative_step=False,scale_parameter=False,warmup_init=False",
        
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
        print("Starting Kaggle TPUv5e-8 Data Streaming run for Transformer Baseline...")
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
