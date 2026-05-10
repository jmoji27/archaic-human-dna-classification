import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """
    Weighted average over sequence positions.
    Learns which positions matter most for the classification decision.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x, mask=None):
        # x: (batch, L, hidden)
        scores = self.attn(x).squeeze(-1)            # (batch, L)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)  # ignore padding positions
        weights = torch.softmax(scores, dim=1)       # (batch, L)
        return (weights.unsqueeze(-1) * x).sum(1)    # (batch, hidden)


class DanqModel(nn.Module):
    """
    Standard DanQ architecture:
      1. Conv layer  — detects local sequence motifs (like a CNN)
      2. MaxPool     — reduces sequence length, keeps strongest signals
      3. BiLSTM      — captures dependencies between motifs across the sequence
      4. Attention   — focuses on the most informative positions
      5. Linear head — classification
    """
    def __init__(self, config, num_classes):
        super().__init__()

        # Conv block 1 — motif detector
        self.conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=config["conv_filters"],
            kernel_size=config["conv_width"],
            padding=config["conv_width"] // 2
        )
        self.bn1 = nn.BatchNorm1d(config["conv_filters"])

        # Conv block 2 — motif combinations
        self.conv2 = nn.Conv1d(
            in_channels=config["conv_filters"],
            out_channels=config["conv_filters"],
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(config["conv_filters"])

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(
            kernel_size=config["pool_size"],
            stride=config["pool_stride"]
        )

        self.dropout_conv = nn.Dropout(config["dropout_conv"])

        # BiLSTM — reads the pooled motif sequence in both directions
        self.lstm = nn.LSTM(
            input_size=config["conv_filters"],
            hidden_size=config["lstm_hidden"],
            num_layers=config["lstm_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=config["dropout_lstm"] if config["lstm_layers"] > 1 else 0.0
        )

        self.dropout_lstm_out = nn.Dropout(config["dropout_lstm"])

        # Attention pool — replaces naive mean/max pool over time
        self.attn_pool = AttentionPool(config["lstm_hidden"] * 2)

        self.dropout = nn.Dropout(config["dropout_dense"])
        self.fc = nn.Linear(config["lstm_hidden"] * 2, num_classes)

    def forward(self, x, mask=None):
        # x: (batch, L, 4)  |  mask: (batch, L) bool, True = padding

        x = x.permute(0, 2, 1)                   # → (batch, 4, L)

        # Conv block 1
        x = self.relu(self.bn1(self.conv1(x)))    # → (batch, filters, L)

        # Conv block 2
        x = self.relu(self.bn2(self.conv2(x)))    # → (batch, filters, L)

        x = self.pool(x)                          # → (batch, filters, L')

        x = x.permute(0, 2, 1)                   # → (batch, L', filters)
        x = self.dropout_conv(x)

        x, _ = self.lstm(x)                       # → (batch, L', hidden*2)
        x = self.dropout_lstm_out(x)

        # Shrink mask to match pooled length L'
        if mask is not None:
            pooled_mask = mask[:, ::self.pool.stride][:, :x.shape[1]]
        else:
            pooled_mask = None

        x = self.attn_pool(x, pooled_mask)        # → (batch, hidden*2)

        x = self.dropout(x)
        x = self.fc(x)
        return x                                  # logits
