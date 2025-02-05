import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    A dataset for language modeling that reads text data from a file,
    tokenizes each line, and creates input-target pairs for training.
    Each sample is a tuple:
        (input_ids, target_ids)
    where:
        - input_ids: tokens for the model input (all tokens except the last)
        - target_ids: tokens shifted one position (all tokens except the first)
    """
    def __init__(self, file_path, tokenizer, max_seq_len=512):
        """
        Args:
            file_path (str): Path to the text file.
            tokenizer: A tokenizer with an `encode` method and optionally `pad_token_id`.
            max_seq_len (int): Maximum sequence length (including special tokens).
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Read and split file into lines
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = f.read().splitlines()
        
        self.samples = []
        pad_token = getattr(tokenizer, "pad_token_id", 0)  # Default to 0 if not defined

        for line in self.lines:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Encode the line into token ids
            token_ids = tokenizer.encode(line)
            
            # Truncate to max_seq_len if necessary
            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]
            
            # We need at least two tokens to create an input-target pair
            if len(token_ids) < 2:
                continue

            # Create input and target pairs
            input_ids = token_ids[:-1]
            target_ids = token_ids[1:]

            # Optionally, pad sequences to a fixed length (max_seq_len - 1)
            # This ensures that each sample has the same length.
            pad_length = (self.max_seq_len - 1) - len(input_ids)
            if pad_length > 0:
                input_ids = input_ids + [pad_token] * pad_length
                target_ids = target_ids + [pad_token] * pad_length

            self.samples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_ids, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    # For testing purposes, assume a simple tokenizer that mimics GPT-2's interface.
    # You should replace this with your actual tokenizer from simple_tokenizer.py.
    class DummyTokenizer:
        def __init__(self):
            # In a real scenario, these values are provided by your tokenizer.
            self.pad_token_id = 0

        def encode(self, text):
            # A dummy encode method: converts characters to their ordinal values.
            # Replace with a real tokenizer (e.g., from Hugging Face's transformers).
            return [ord(c) for c in text]

        def decode(self, token_ids):
            # A dummy decode method: converts ordinal values back to characters.
            return ''.join([chr(tid) for tid in token_ids if tid != self.pad_token_id])
    
    # Create a dummy tokenizer instance for testing
    tokenizer = DummyTokenizer()
    
    # Create a small test file if it doesn't exist
    test_file = "test_texts.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("the quick brown fox jumps over the lazy dog\n")
        f.write("once upon a time in a magical forest\n")
    
    # Initialize the dataset with the test file
    dataset = TextDataset(test_file, tokenizer, max_seq_len=20)
    print(f"Total samples: {len(dataset)}")
    
    # Display a sample from the dataset
    sample_input, sample_target = dataset[0]
    print("Sample input token ids:", sample_input)
    print("Sample target token ids:", sample_target)
