import torch
import torch.nn as nn
from utils import patchify_batch
from einops.layers.torch import Rearrange

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
    def forward(self, x):
        return self.proj(x)

class CFDFeatureEmbedder(nn.Module):
    """
    Transforms raw features (time, obstacle, charge) into tokens
    of shape [batch_size, embed_dim].
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        """
        x: shape [batch_size, input_dim] 
        returns: shape [batch_size, embed_dim]
        """
        #print(f'x shape: {x.shape}')
        return self.projection(x)

class BlastFormer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, seq_len, patch_size):
        super().__init__()
        # Initilize feature embedding layers
        self.wall_embedder = CFDFeatureEmbedder(6, hidden_dim)
        self.charge_embedder = CFDFeatureEmbedder(7, hidden_dim)
        self.time_embedder = CFDFeatureEmbedder(1, hidden_dim)
        


        self.seq_len = seq_len
        self.patch_size = patch_size

        # Input projection to match transformer hidden dimension
        #self.patch_proj = nn.Linear(input_dim, hidden_dim)
        self.patch_proj = PatchEmbed(1, hidden_dim, patch_size)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Output projection to predict pressures
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, pressure, charge_data, wall_locations, time):
        """
        Args:
            src: Input tensor (encoder input) of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Embed features
        wall_embedded = self.wall_embedder(wall_locations)
        charge_embedded = self.charge_embedder(charge_data)
        time_embedded = self.time_embedder(time)
        #patch = patchify_batch(pressure.squeeze(1), self.patch_size)
        #projected_patch = self.patch_proj(patch)
        projected_patch = self.patch_proj(pressure.unsqueeze(1))



        # Combine features
        src = torch.cat([projected_patch, charge_embedded.unsqueeze(1), wall_embedded, time_embedded.unsqueeze(1)], dim=1)
        # pass through encoder
        output = self.encoder(src)

        # Project to output dimension
        output = self.output_proj(output)


        return output
