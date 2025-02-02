
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
# Simple Tokenizer
class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_idx = {word: i for i, word in enumerate(vocab)}
        self.idx_to_token = {i: word for i, word in enumerate(vocab)}
    
    def encode(self, text):
        return [self.token_to_idx[word] for word in text.split() if word in self.token_to_idx]
    
    def decode(self, tokens):
        return ' '.join([self.idx_to_token[token] for token in tokens])