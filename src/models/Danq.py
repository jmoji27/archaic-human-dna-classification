import torch
import torch.nn as nn
import torch.nn.functional as F


class DanqModel(nn.Module):
    """
    DanQ adapted for short ancient DNA reads (35-85bp).

    Original paper used:
    - Conv: 320 kernels, width 26, for 1000bp sequences
    - Pool: size 13, stride 13
    - BiLSTM: 320 hidden units

    Scaled down here for 35-85bp sequences:
    - Smaller kernels (won't fit 26bp kernel on 35bp sequence)
    - Smaller pool (can't pool by 13 on short sequences)
    - Smaller LSTM (less sequence to model)
    """

    def __init__(self, config, num_classes):
        super(DanqModel, self).__init__()

        conv_filters    = config["conv_filters"]
        conv_width      = config["conv_width"]
        pool_size       = config["pool_size"]
        pool_stride     = config["pool_stride"]
        lstm_hidden     = config["lstm_hidden"]
        lstm_layers     = config["lstm_layers"]
        dropout_conv    = config["dropout_conv"]
        dropout_lstm    = config["dropout_lstm"]
        dropout_dense   = config["dropout_dense"]
        dense_units     = config["dense_units"]

        # ── Conv motif scanner ─────────────────────────────────────────────
        self.conv = nn.Conv1d(
            in_channels=4,
            out_channels=conv_filters,
            kernel_size=conv_width,
            padding=conv_width // 2,  # same padding so short seqs survive
        )
        self.bn        = nn.BatchNorm1d(conv_filters)
        self.pool      = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.drop_conv = nn.Dropout(dropout_conv)

        # ── BiLSTM grammar learner ─────────────────────────────────────────
        # batch_first=True → input/output shape: (batch, seq, features)
        self.bilstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_lstm if lstm_layers > 1 else 0.0,
        )
        self.drop_lstm = nn.Dropout(dropout_lstm)

        # ── Dense head ────────────────────────────────────────────────────
        # BiLSTM output is lstm_hidden * 2 (forward + backward)
        # We global-average-pool over the time dimension first
        self.fc1       = nn.Linear(lstm_hidden * 2, dense_units)
        self.drop_dense = nn.Dropout(dropout_dense)
        self.output    = nn.Linear(dense_units, num_classes)

    def forward(self, x):
        # x: (batch, L, 4)
        x = x.permute(0, 2, 1)          # → (batch, 4, L)

        # Conv block
        x = self.conv(x)                 # (batch, filters, L)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)                 # (batch, filters, L/pool)
        x = self.drop_conv(x)

        # BiLSTM expects (batch, seq_len, features)
        x = x.permute(0, 2, 1)          # → (batch, L/pool, filters)
        x, _ = self.bilstm(x)           # → (batch, L/pool, lstm_hidden*2)
        x = self.drop_lstm(x)

        # Global average pool over time → (batch, lstm_hidden*2)
        x = x.mean(dim=1)

        # Dense head
        x = F.relu(self.fc1(x))
        x = self.drop_dense(x)
        x = self.output(x)
        return x  # raw logits — CrossEntropyLoss handles softmax