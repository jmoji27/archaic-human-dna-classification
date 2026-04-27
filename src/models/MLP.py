import torch
import torch.nn as nn

class MLPKmer(nn.Module):
    """
    Simple MLP for k-mer frequency features.
    
    Architecture:
        Input (vocab_size) -> FC -> ReLU -> Dropout -> FC -> ReLU -> Dropout -> Output
    """
    
    def __init__(self, config, num_classes):
        super().__init__()
        
        vocab_size = config["vocab_size"]
        hidden_dims = config.get("hidden_dims", [256, 128])
        dropout = config.get("dropout", 0.3)
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(vocab_size, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch, vocab_size) - k-mer frequency features
        Returns:
            (batch, num_classes) - logits
        """
        return self.model(x)