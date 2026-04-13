import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.embedding = nn.Linear(4, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, L, 4)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x