import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    def __init__(self, config, num_classes):
        super(CNN1D, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        in_channels = 4  # DNA one-hot (A,C,G,T)

        #  Conv stack 
        for i in range(config["num_conv_layers"]):
            out_channels = config["conv_filters"][i] if isinstance(config["conv_filters"], list) else config["conv_filters"]
            kernel_size = config["conv_width"][i] if isinstance(config["conv_width"], list) else config["conv_width"]
            stride = config["conv_stride"][i] if isinstance(config["conv_stride"], list) else config["conv_stride"]
            dropout = config["dropout_rate_conv"][i] if isinstance(config["dropout_rate_conv"], list) else config["dropout_rate_conv"]

            conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
            bn = nn.BatchNorm1d(out_channels)

            self.conv_layers.append(conv)
            self.bn_layers.append(bn)
            self.dropout_layers.append(nn.Dropout(dropout))

            in_channels = out_channels

        #  Pooling 
        self.pool = nn.MaxPool1d(
            kernel_size=config["max_pool_size"],
            stride=config["max_pool_stride"]
        )

        # --- Global pooling (replaces Flatten) ---
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # --- Dense stack ---
        self.fc_layers = nn.ModuleList()
        self.fc_dropout = nn.ModuleList()

        in_features = in_channels

        for i in range(config["num_dense_layers"]):
            out_features = config["dense_filters"][i] if isinstance(config["dense_filters"], list) else config["dense_filters"]
            dropout = config["dropout_rate_dense"][i] if isinstance(config["dropout_rate_dense"], list) else config["dropout_rate_dense"]

            self.fc_layers.append(nn.Linear(in_features, out_features))
            self.fc_dropout.append(nn.Dropout(dropout))

            in_features = out_features

        # --- Output ---
        self.output = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Input expected: (batch, L, 4)
        x = x.permute(0, 2, 1)  # → (batch, 4, L)

        # Conv stack
        for conv, bn, drop in zip(self.conv_layers, self.bn_layers, self.dropout_layers):
            x = conv(x) #scan for motifs
            x = bn(x) #normalize
            x = F.relu(x) #non_linearity
            x = drop(x) # regularization
            x = self.pool(x)  # pooling after each block

        # Global pooling
        x = self.global_pool(x)  # (batch, C, 1)
        x = x.squeeze(-1)        # (batch, C)

        # Dense stack
        for fc, drop in zip(self.fc_layers, self.fc_dropout):
            x = F.relu(fc(x))
            x = drop(x)

        # Output
        x = self.output(x)

        return x  # use CrossEntropyLoss (no softmax here)


