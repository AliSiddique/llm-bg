import torch
import torch.nn.functional as F
from model import TransformerModel
from config import config
from simple_tokenizer import get_tokenizer
import math

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    
    Args:
        logits (Tensor): Logits distribution of shape (..., vocabulary_size)
        top_k (int): Keep only top k tokens with highest probability (top-k filtering).
        top_p (float): Keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        filter_value (float): Logits below the threshold are set to this value.
        
    Returns:
        Tensor: Filtered logits.
    """
    # Top-K filtering
    if top_k > 0:
        # Remove all tokens with a probability less than the top-k tokens
        top_k = min(top_k, logits.size(-1))  # Safety check
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        # Sort logits in descending order to calculate cumulative probabilities
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted indices to original ordering
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=40, top_p=0.9, device='cpu'):
    """
    Generates text using the model given a prompt, with support for top-k and top-p sampling.
    
    Args:
        model (nn.Module): The Transformer language model.
        tokenizer: The tokenizer with encode/decode methods.
        prompt (str): The input prompt string.
        max_length (int): Maximum number of tokens to generate.
        temperature (float): Temperature for scaling logits.
        top_k (int): The top k tokens to consider for sampling.
        top_p (float): The nucleus (top-p) probability threshold.
        device (str): Device to run the generation on.
        
    Returns:
        str: The generated text.
    """
    model.eval()
    
    # Encode the prompt and move to device
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits from the model; shape: [batch_size, seq_length, vocab_size]
            logits = model(input_ids)
            # Focus on the logits for the last generated token: shape: [batch_size, vocab_size]
            next_token_logits = logits[:, -1, :]
            
            # Scale logits by temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k and top-p filtering
            filtered_logits = top_k_top_p_filtering(next_token_logits.clone(), top_k=top_k, top_p=top_p)
            probabilities = F.softmax(filtered_logits, dim=-1)
            
            # Sample one token from the filtered distribution
            next_token = torch.multinomial(probabilities, num_samples=1)
            
            # Append sampled token to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # (Optional) Stop if the EOS token is generated (if your tokenizer defines one)
            if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the full sequence of token ids into text
    return tokenizer.decode(input_ids[0].tolist())

if __name__ == '__main__':
    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer (from your simple_tokenizer.py)
    tokenizer = get_tokenizer()
    # Update vocab size based on tokenizer (if applicable)
    vocab_size = len(tokenizer)
    
    # Initialize the model using parameters from config.py
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_heads=config.n_head,
        num_layers=config.num_layers,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len
    ).to(device)
    
    print("Interactive Mode: Type your prompt and press Enter. Type 'quit' to exit.\n")
    
    # Interactive loop to converse with the model
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["quit", "exit"]:
            break
        
        # Generate text response from the model
        generated_text = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_length=50, 
            temperature=config.temperature, 
            top_k=config.top_k, 
            top_p=0.9,  # You can adjust this value as needed
            device=device
        )
        print("LLM:", generated_text, "\n")
