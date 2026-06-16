import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Import your model (ensure it's in the same directory)
from cloud_gpu_streaming_causal_fourier import ContinuousFourierLM

def evaluate_lambada(model_path, device="cuda"):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    print("Loading model...")
    model = ContinuousFourierLM(vocab_size=len(tokenizer), latent_dim=768, num_layers=12, num_modes=128)
    
    # Load weights - adjust path as needed
    # state_dict = torch.load(model_path, map_location=device)
    # model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Loading LAMBADA dataset...")
    # EleutherAI/lambada_openai is the standard split used for evaluating GPT-style models
    dataset = load_dataset("EleutherAI/lambada_openai", split="test")

    correct = 0
    total = len(dataset)

    print("Evaluating...")
    with torch.no_grad():
        for example in tqdm(dataset, total=total):
            text = example["text"]
            
            # The task in LAMBADA is to predict the final word.
            # We split the text into "context" and the "target word".
            words = text.strip().split()
            if len(words) < 2:
                continue
                
            context_words = words[:-1]
            target_word = words[-1]
            
            context_text = " ".join(context_words)
            
            # Tokenize context
            input_ids = tokenizer(context_text, return_tensors="pt").input_ids.to(device)
            
            # Generate the next tokens
            # We need to generate enough tokens to cover the target word.
            # BERT might split the target word into multiple subwords (e.g., "apple" -> "ap", "##ple").
            # A safe bet is to generate a few tokens and decode them.
            
            generated_ids = input_ids[0].tolist()
            
            # Generate up to 3 tokens (usually enough for one word in BERT)
            for _ in range(3):
                inputs = torch.tensor([generated_ids]).to(device)
                outputs = model(inputs)
                next_token_logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
                generated_ids.append(next_token)
                
                # If we hit a space or punctuation that indicates the end of a word, we can stop
                decoded_so_far = tokenizer.decode(generated_ids[len(input_ids[0]):])
                if len(decoded_so_far.strip()) >= len(target_word) or " " in decoded_so_far.strip():
                    break
                    
            # Extract the newly generated string
            generated_text = tokenizer.decode(generated_ids[len(input_ids[0]):]).strip().lower()
            target_word_lower = target_word.lower()
            
            # Check if the generated text starts with the target word
            # (or if the target word starts with the generated text in case of early stopping)
            if generated_text.startswith(target_word_lower) or target_word_lower.startswith(generated_text):
                correct += 1

    accuracy = (correct / total) * 100
    print(f"\nLAMBADA Accuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    # Replace with your actual checkpoint path
    evaluate_lambada("path_to_your_checkpoint.pt")
