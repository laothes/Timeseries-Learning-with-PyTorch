import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # [batch_size, sequence_length] -> [batch_size, sequence_length, d_model]
        return self.embed(x)


class LearningPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model * 2, d_model)

        pe = torch.zeros(max_seq_len, d_model)  # [max_seq_len, d_model]
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # [max_seq_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).unsqueeze(
            0)  # [1, d_model/2]
        pe[:, 0::2] = torch.sin(pos @ div_term)  # even position
        pe[:, 1::2] = torch.cos(pos @ div_term)  # odd position
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, ax_seq_len, d_model]

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        pe = self.pe[:, :seq_len, :].expand(batch_size, -1, -1)  # [batch_size, seq_len, d_model]
        x = self.linear(torch.cat((x, pe), dim=-1))
        return self.dropout(x)

class SelfAttention(nn.Module): pass

class MultiHeadAttention(nn.Module):



if __name__ == '__main__': pass
