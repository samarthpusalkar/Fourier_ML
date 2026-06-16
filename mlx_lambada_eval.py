import os
import argparse
import mlx.core as mx
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Import the native MLX model you already wrote!
from mlx_inference import ContinuousFourierLM

def evaluate_lambada_mlx(repo_id=None, checkpoint_path=None):
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    print("Initializing Native MLX Model...")
    model = ContinuousFourierLM(vocab_size=len(tokenizer), latent_dim=768, num_layers=12, num_modes=128)
    
    # ---------------------------------------------------------
    # WEIGHT LOADING LOGIC (Mirrors mlx_inference.py)
    # ---------------------------------------------------------
    if not checkpoint_path and repo_id:
        print(f"Fetching latest checkpoint from Hugging Face Repo: {repo_id}...")
        try:
            from huggingface_hub import hf_hub_download, login
            login(os.getenv('HF_TOKEN'))
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
            print(f"Downloaded securely to: {checkpoint_path}")
        except Exception as e:
            print(f"\n[!] Error fetching from HF Hub: {e}")
            checkpoint_path = "./CausalFourierLM_Checkpoints_BiggerDataset_GPU/best_model_streaming/model.safetensors"

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        weights = mx.load(checkpoint_path)
        new_weights = []
        
        for k, v in weights.items():
            if k.startswith("model."):
                k = k[6:] # Strip HF wrapper
            # Map PyTorch `ffn.0` to MLX `ffn.layers.0`
            if ".ffn." in k:
                k = k.replace(".ffn.0.", ".ffn.layers.0.")
                k = k.replace(".ffn.1.", ".ffn.layers.1.")
                k = k.replace(".ffn.3.", ".ffn.layers.3.")
                
            new_weights.append((k, v))
            
        model.load_weights(new_weights, strict=False)
        print("✅ Weights successfully mapped to Apple Silicon Unified Memory!")
    else:
        print("⚠️ Warning: Could not find any checkpoint. Evaluating with random weights.")
        
    # Flush initialization
    mx.eval(model.parameters())

    # ---------------------------------------------------------
    # LAMBADA EVALUATION LOOP
    # ---------------------------------------------------------
    print("\nLoading LAMBADA dataset...")
    dataset = load_dataset("EleutherAI/lambada_openai", split="test")

    correct = 0
    total = len(dataset)

    print("Evaluating on Apple Silicon...")
    for example in tqdm(dataset, total=total):
        text = example["text"]
        
        # LAMBADA target is the final word
        words = text.strip().split()
        if len(words) < 2:
            continue
            
        context_words = words[:-1]
        target_word = words[-1]
        
        context_text = " ".join(context_words)
        
        # Tokenize context
        input_ids = tokenizer.encode(context_text, add_special_tokens=False)
        original_length = len(input_ids)
        
        generated_ids = input_ids.copy()
        
        # Generate up to 3 tokens to cover sub-word splits (e.g., "apple" -> "ap", "##ple")
        for _ in range(3):
            # Convert to MLX array (Batch size 1)
            x = mx.array([generated_ids])
            
            # Forward pass
            logits = model(x)
            
            # Get the logits for the last token in the sequence
            next_token_logits = logits[0, -1, :]
            
            # Greedy decoding (argmax)
            next_token = mx.argmax(next_token_logits).item()
            generated_ids.append(next_token)
            
            # Decode just the newly generated part to see if we completed the word
            decoded_so_far = tokenizer.decode(generated_ids[original_length:])
            if len(decoded_so_far.strip()) >= len(target_word) or " " in decoded_so_far.strip():
                break
                
        # Extract the newly generated string
        generated_text = tokenizer.decode(generated_ids[original_length:]).strip().lower()
        target_word_lower = target_word.lower()
        
        # Check if the generated text matches the target word
        if generated_text.startswith(target_word_lower) or target_word_lower.startswith(generated_text):
            correct += 1

    accuracy = (correct / total) * 100
    print(f"\nLAMBADA Accuracy: {accuracy:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="CodeIsAbstract/Fourier_LM_Checkpoints_Continued")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    evaluate_lambada_mlx(repo_id=args.repo_id, checkpoint_path=args.checkpoint)
