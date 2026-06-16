import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
