"""
run_inference.py — Dynamic MLM Inference for Spectral Architectures
===================================================================
A single script capable of dynamically loading and running inference 
on any of the Spectral Language Model versions (scaled, tied, fractional).
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
import sys
import os
from safetensors.torch import load_file

def load_architecture(arch, vocab_size, seq_length=64, latent_dim=None, num_modes=None, num_layers=None, resolution=None):
    """Dynamically imports and initializes the correct architecture."""
    if arch == "scaled":
        from scaled_spectral_language_model import LargeSpectralLM
        model = LargeSpectralLM(
            vocab_size=vocab_size,
            seq_length=seq_length,
            latent_dim=latent_dim if latent_dim is not None else 128,
            num_modes=num_modes if num_modes is not None else 64,
            num_layers=num_layers if num_layers is not None else 4
        )
        ckpt_path = "results/best_scaled_spectral.pt"
        
    elif arch == "tied":
        from tied_spectral_language_model import LargeSpectralLM
        model = LargeSpectralLM(
            vocab_size=vocab_size,
            seq_length=seq_length,
            latent_dim=latent_dim if latent_dim is not None else 512,
            num_modes=num_modes if num_modes is not None else 64,
            num_layers=num_layers if num_layers is not None else 6
        )
        ckpt_path = "results/best_tied_spectral.pt"
        
    elif arch == "fractional":
        from fractional_spectral_language_model import LargeSpectralLM
        model = LargeSpectralLM(
            vocab_size=vocab_size,
            seq_length=seq_length,
            latent_dim=latent_dim if latent_dim is not None else 2048,
            num_modes=num_modes if num_modes is not None else 64,
            num_layers=num_layers if num_layers is not None else 7,
            resolution=resolution if resolution is not None else 0.05
        )
        ckpt_path = "results/best_fractional_spectral.pt"
        
    else:
        raise ValueError(f"Unknown architecture: {arch}")
        
    return model, ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Spectral MLM Inference")
    parser.add_argument("--arch", type=str, default="scaled", choices=["scaled", "tied", "fractional"],
                        help="Which architecture version to load")
    parser.add_argument("--seq_length", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=None, help="Override default latent dim")
    parser.add_argument("--num_modes", type=int, default=None, help="Override default num modes")
    parser.add_argument("--num_layers", type=int, default=None, help="Override default num layers")
    parser.add_argument("--resolution", type=float, default=None, help="Override fractional resolution")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to a specific checkpoint directory or file")
    args = parser.parse_args()

    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Inference Environment on: {device}")

    # Load tokenizer
    print("Loading HuggingFace Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = len(tokenizer)

    # Initialize model
    print(f"Initializing '{args.arch}' architecture...")
    model, ckpt_path = load_architecture(
        args.arch, 
        vocab_size, 
        args.seq_length,
        args.latent_dim,
        args.num_modes,
        args.num_layers,
        args.resolution
    )
    
    
    # Override default checkpoint path if user provided one
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
        
    if not os.path.exists(ckpt_path):
        print(f"\n[ERROR] Checkpoint not found at {ckpt_path}!")
        print("Please ensure the model has finished at least 1 epoch and saved best weights.")
        sys.exit(1)
        
    print(f"Loading weights from {ckpt_path}...")
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path, device=str(device))
    elif os.path.isdir(ckpt_path) and os.path.exists(os.path.join(ckpt_path, "model.safetensors")):
        state_dict = load_file(os.path.join(ckpt_path, "model.safetensors"), device=str(device))
    else:
        state_dict = torch.load(ckpt_path, map_location=device)
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("\n" + "="*60)
    print("SPECTRAL MASKED LANGUAGE MODEL INTERACTIVE INFERENCE")
    print("="*60)
    print("Type a sentence and include the exact word [MASK] where")
    print("you want the model to predict the missing word.")
    print("Example: The quick brown [MASK] jumps over the lazy dog.")
    print("Type 'q' or 'quit' to exit.")
    print("="*60 + "\n")

    while True:
        text = input("\nInput: ")
        if text.lower() in ["q", "quit", "exit"]:
            break
            
        if "[MASK]" not in text:
            print("[WARNING] You must include the literal text '[MASK]' in your sentence.")
            continue
            
        # Tokenize
        tokens = tokenizer(text, return_tensors="pt", max_length=args.seq_length, truncation=True, padding="max_length")
        input_ids = tokens["input_ids"].to(device)
        
        # Find mask index
        mask_token_id = tokenizer.mask_token_id
        mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]
        
        if len(mask_indices) == 0:
            print("[WARNING] Could not find the mask token after tokenization. Is the sentence too long?")
            continue
            
        with torch.no_grad():
            output = model(input_ids)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                # HF Trainer MaskedLMOutput
                logits = output.logits
            
        # Iterate over all masks found (usually just 1)
        for idx in mask_indices:
            # logits shape: (1, seq_length, vocab_size)
            mask_logits = logits[0, idx, :]
            probs = F.softmax(mask_logits, dim=0)
            
            top_k = 5
            top_probs, top_indices = torch.topk(probs, top_k)
            
            print(f"\nPredictions for [MASK] at position {idx.item()}:")
            print("-" * 40)
            for i in range(top_k):
                word = tokenizer.decode([top_indices[i].item()]).strip()
                p = top_probs[i].item() * 100
                
                # Draw a little text bar chart
                bar_len = int(p / 2) # max 50 chars for 100%
                bar = "█" * bar_len
                print(f"{i+1}. {word:<15} | {p:5.2f}% | {bar}")

if __name__ == "__main__":
    main()
