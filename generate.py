# Text Generation
def generate(model, tokenizer, prompt, max_length=50):
    model.eval()
    tokens = tokenizer.encode(prompt)
    output = tokens[:]
    
    for _ in range(max_length):
        input_tensor = torch.tensor([output], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_tensor)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            output.append(next_token)
            if next_token == tokenizer.token_to_idx.get('<EOS>', None):
                break
    
    return tokenizer.decode(output)
