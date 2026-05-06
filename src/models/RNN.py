import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, config, num_classes):
        super(RNNModel, self).__init__()
        
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.input_size = 4
        
        self.lstm = nn.LSTM(
            self.input_size, 
            self.hidden_size, 
            self.num_layers, 
            batch_first=True, 
            dropout=config.get('dropout', 0.3),
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.3)),
            nn.Linear(self.hidden_size, num_classes)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        
        final_hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        
        return self.fc(final_hidden)