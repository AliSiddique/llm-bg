from transformers import GPT2Tokenizer

def get_tokenizer():
    """
    Loads and returns the GPT-2 tokenizer.
    Ensures that the tokenizer has a pad_token (required for batching).
    """
    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Ensure a pad token exists; if not, add one.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    return tokenizer

if __name__ == '__main__':
    # Test the tokenizer with a sample text.
    tokenizer = get_tokenizer()
    
    test_text = "Hello, how are you today?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print("Original text:", test_text)
    print("Encoded token IDs:", encoded)
    print("Decoded text:", decoded)
