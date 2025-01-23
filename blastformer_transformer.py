import torch
import torch.nn as nn

class PressurePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, seq_len):
        super(PressurePredictor, self).__init__()
        self.seq_len = seq_len

        # Input projection to match transformer hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Transformer Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # Output projection to predict pressures
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tgt):
        """
        Args:
            src: Input tensor (encoder input) of shape (batch_size, seq_len, input_dim)
            tgt: Target tensor (decoder input) of shape (batch_size, seq_len, output_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Project inputs to hidden dimension
        src = self.input_proj(src)  # Shape: (batch_size, seq_len, hidden_dim)
        tgt = self.input_proj(tgt)  # Shape: (batch_size, seq_len, hidden_dim)

        # Encode input features
        memory = self.encoder(src)  # Shape: (batch_size, seq_len, hidden_dim)

        # Decode target using encoded memory
        output = self.decoder(tgt, memory)  # Shape: (batch_size, seq_len, hidden_dim)

        # Project output to predict pressures
        output = self.output_proj(output)  # Shape: (batch_size, seq_len, output_dim)
        return output
