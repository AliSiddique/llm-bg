



# Dataset Loader
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=50):
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.texts[idx])
        input_ids = encoded[:-1]
        target_ids = encoded[1:]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)