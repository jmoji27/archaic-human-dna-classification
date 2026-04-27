import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.hidden_size = 64 
        self.num_layers = 1
        dropout_rate = 0.5
        
        self.embedding = nn.Embedding(num_embeddings=4, embedding_dim=32)

        self.rnn = nn.LSTM(
            input_size=32,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if self.num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x_indices = torch.argmax(x, dim=-1)
        x_embedded = self.embedding(x_indices) 
        
        out, _ = self.rnn(x_embedded)
        out = torch.mean(out, dim=1)
        out = self.fc(out)
        
        return out