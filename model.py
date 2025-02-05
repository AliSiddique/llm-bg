import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements the standard positional encoding as described in 
    "Attention Is All You Need" (Vaswani et al. 2017).
    """
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Pre-compute positional encodings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # If d_model is odd, handle the last column separately.
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_length, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        d_model=256, 
        num_heads=8, 
        num_layers=4, 
        dropout=0.1,
        max_seq_len=512
    ):
        super(TransformerModel, self).__init__()
        self.d_model = d_model

        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer Encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True  # This makes the expected input shape [batch_size, seq_length, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection layer to map transformer output to vocabulary logits
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.zeros_(self.fc_out.bias)
        nn.init.normal_(self.fc_out.weight, mean=0, std=self.d_model ** -0.5)

    def forward(self, input_ids):
        """
        input_ids: [batch_size, seq_length]
        """
        # Get token embeddings and scale them by sqrt(d_model)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)  # [batch_size, seq_length, d_model]
        # Add positional encoding
        x = self.pos_encoder(x)  # [batch_size, seq_length, d_model]
        # Pass through the transformer encoder (no need to transpose because of batch_first)
        x = self.transformer_encoder(x)
        # Project to vocabulary logits: [batch_size, seq_length, vocab_size]
        logits = self.fc_out(x)
        return logits

if __name__ == "__main__":
    # For testing purposes, use a small vocab size and dummy input
    vocab_size = 10  # Example vocab size
    model = TransformerModel(vocab_size=vocab_size, d_model=256, num_heads=8, num_layers=4, dropout=0.1, max_seq_len=20)
    
    # Dummy input: [batch_size, seq_length] with random token indices
    dummy_input = torch.randint(0, vocab_size, (2, 10))
    print("Dummy input:", dummy_input)
    
    # Forward pass through the model
    output = model(dummy_input)
    print("Model output shape:", output.shape)  # Expected: [2, 10, vocab_size]
