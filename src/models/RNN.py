import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=4,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, L, 4)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out