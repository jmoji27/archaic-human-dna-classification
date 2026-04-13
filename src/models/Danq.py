import torch
import torch.nn as nn
import torch.nn.functional as F


class DanqModel(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.conv = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, L, 4)
        x = x.permute(0, 2, 1)       # (batch, 4, L)
        x = F.relu(self.conv(x))
        x = self.pool(x)

        x = x.permute(0, 2, 1)       # (batch, L, C)

        x, _ = self.lstm(x)
        x = x[:, -1, :]

        x = self.fc(x)
        return x