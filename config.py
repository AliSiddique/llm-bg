# config.py
class Config:
    # Model architecture
    vocab_size = 50257  # GPT-2 vocab size
    d_model = 768       # Embedding dimension
    n_head = 12         # Number of attention heads
    num_layers = 6      # Number of transformer layers
    dropout = 0.1       # Dropout rate
    max_seq_len = 512   # Maximum sequence length
    
    # Training parameters
    batch_size = 1
    learning_rate = 3e-5
    epochs = 10
    
    # Dataset
    dataset_path = "test_texts.txt"
    
    # Generation
    temperature = 0.7
    top_k = 40

# Create config instance
config = Config()