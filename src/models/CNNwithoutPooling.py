import torch
import torch.nn as nn
import torch.nn.functional as F


def _get(cfg, key, i):
    """Get config value — handles both list-per-layer and scalar."""
    v = cfg[key]
    return v[i] if isinstance(v, list) else v


class CNN1D_no_pooling(nn.Module):
    def __init__(self, config, num_classes, seq_length, strict_length=True):
        super().__init__()
        self.seq_length = seq_length
        self._fc_built = False

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
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

        self.final_channels = in_channels
        self.fc_layers  = nn.ModuleList()
        self.fc_dropout = nn.ModuleList()
        self._fc_config = config
        self._num_dense = config["num_dense_layers"]
        self._num_classes = num_classes
        self.output_layer = None

    def _build_fc(self, in_features):
        """Called once on first forward pass with the real flattened size."""
        config = self._fc_config
        device = next(self.conv_layers[0].parameters()).device

        for i in range(self._num_dense):
            out_features = _get(config, "dense_filters", i)
            self.fc_layers.append(
                nn.Linear(in_features, out_features).to(device)
            )
            self.fc_dropout.append(
                nn.Dropout(_get(config, "dropout_rate_dense", i))
            )
            in_features = out_features

        self.output_layer = nn.Linear(in_features, self._num_classes).to(device)
        self._fc_built = True

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

        if not self._fc_built:
            self._build_fc(x.size(1))

        for fc, drop in zip(self.fc_layers, self.fc_dropout):
            x = F.relu(fc(x))
            x = drop(x)

        return self.output_layer(x)



    