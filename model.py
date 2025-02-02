import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json

# Transformer Model
def get_model(vocab_size, d_model=256, num_heads=8, num_layers=6, dropout=0.1):
    from torch.nn import Transformer
    return Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout)
