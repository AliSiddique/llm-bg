import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TransformerModel
from config import config
from dataset import TextDataset  # Make sure this file implements the dataset class
from simple_tokenizer import get_tokenizer

def train():
    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer (from your simple_tokenizer.py)
    tokenizer = get_tokenizer()
    # Update vocab size based on tokenizer (if applicable)
    vocab_size = len(tokenizer)

    # Initialize the dataset and dataloader
    dataset = TextDataset(config.dataset_path, tokenizer, max_seq_len=config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    
    # Initialize the model using parameters from config.py
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_heads=config.n_head,
        num_layers=config.num_layers,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len
    ).to(device)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Note: for language modeling, target tokens are shifted by one position.
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    model.train()  # Set model to training mode
    
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            # Move data to device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: model returns logits of shape [batch_size, seq_length, vocab_size]
            logits = model(input_ids)
            
            # Reshape logits and targets for calculating cross-entropy loss
            # Flatten the sequence: (batch_size * seq_length, vocab_size) and (batch_size * seq_length)
            logits = logits.view(-1, logits.size(-1))
            targets = target_ids.view(-1)
            
            loss = criterion(logits, targets)
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch}/{config.epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"==> Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # (Optional) Save the model checkpoint after each epoch
        torch.save(model.state_dict(), f"transformer_epoch_{epoch}.pt")
    
    print("Training complete.")

if __name__ == "__main__":
    train()
