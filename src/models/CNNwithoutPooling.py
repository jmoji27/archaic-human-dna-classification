import torch
import torch.nn as nn
import torch.nn.functional as F


def _get(cfg, key, i):
    """Get config value — handles both list-per-layer and scalar."""
    v = cfg[key]
    return v[i] if isinstance(v, list) else v


class CNN1D_no_pooling(nn.Module):
    def __init__(self, config, num_classes, seq_length):
        super().__init__()
        self.seq_length = seq_length

        # ── conv layers ──
        self.conv_layers    = nn.ModuleList()
        self.bn_layers      = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        in_channels = 4  # A, C, G, T one-hot
        for i in range(config["num_conv_layers"]):
            out_channels = _get(config, "conv_filters", i)
            kernel_size  = _get(config, "conv_width", i)
            dropout      = _get(config, "dropout_rate_conv", i)

            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=kernel_size,
                          stride=1,
                          padding="same")
            )
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            self.dropout_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.final_channels = in_channels  # 32 with current config

        # ── FC layers built eagerly — seq_length is known now ──
        fc_in = seq_length * self.final_channels  # e.g. 85 * 32 = 2720

        self.fc_layers  = nn.ModuleList()
        self.fc_dropout = nn.ModuleList()

        for i in range(config["num_dense_layers"]):
            out_features = _get(config, "dense_filters", i)
            self.fc_layers.append(nn.Linear(fc_in, out_features))
            self.fc_dropout.append(nn.Dropout(_get(config, "dropout_rate_dense", i)))
            fc_in = out_features

        self.output_layer = nn.Linear(fc_in, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, 4, L)

        for conv, bn, drop in zip(self.conv_layers, self.bn_layers, self.dropout_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = drop(x)

        # Force exactly seq_length — crop if over, pad if under
        if x.size(2) > self.seq_length:
            x = x[:, :, :self.seq_length]
        elif x.size(2) < self.seq_length:
            pad = torch.zeros(x.size(0), x.size(1),
                              self.seq_length - x.size(2), device=x.device)
            x = torch.cat([x, pad], dim=2)

        x = x.reshape(x.size(0), -1)

        for fc, drop in zip(self.fc_layers, self.fc_dropout):
            x = F.relu(fc(x))
            x = drop(x)

        return self.output_layer(x)
    