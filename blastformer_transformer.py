import torch
import torch.nn as nn

class PressurePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, seq_len):
        super(PressurePredictor, self).__init__()
        self.seq_len = seq_len

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.output_proj(x)
        return x