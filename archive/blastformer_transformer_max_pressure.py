import torch
import torch.nn as nn
from utils import patchify_batch, unpatchify_batch
from einops.layers.torch import Rearrange
from hdf5_dataset_new import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


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
    def __init__(self, wall_dim, charge_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        # Initilize feature embedding layers
        self.wall_embedder = CFDFeatureEmbedder(wall_dim, hidden_dim)
        self.charge_embedder = CFDFeatureEmbedder(charge_dim, hidden_dim)
        
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Output projection to predict pressures
        self.output_proj_1 = nn.Linear(hidden_dim, output_dim)
        self.output_proj_2 = nn.Linear(output_dim*2, output_dim)


    def forward(self, charge_data, wall_locations):
        """
        Args:
            src: Input tensor (encoder input) of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Embed features
        wall_embedded = self.wall_embedder(wall_locations)
        charge_embedded = self.charge_embedder(charge_data)



        # Combine features
        src = torch.cat([charge_embedded.unsqueeze(1), wall_embedded.unsqueeze(1)], dim=1)
        # pass through encoder
        output = self.encoder(src)

        # Project to output dimension
        output = self.output_proj_1(output)
        # reshape from [batch_size, 2, output_dim] to [batch_size, output_dim*2]
        output = output.reshape(output.shape[0], -1)
        output = self.output_proj_2(output)
        reconstructed_pressure = output.reshape(output.shape[0], 99, 99)
        return reconstructed_pressure
    
    
def main():
    input_dim = (99**2)//(11**2)
    hidden_dim = 256
    output_dim = input_dim 
    seq_len = 302 # Context timesteps
    num_layers = 4
    patch_size = 3
    max_samples_to_process = 100

    model = BlastFormer(input_dim, hidden_dim, num_layers, output_dim, seq_len, patch_size)
    print(model)
    dataset = BlastDataset("/home/reid/projects/blast_waves/hdf5_dataset", split="test", normalize=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    original_pressures = []
    reconstructed_pressures = []

    for batch in dataloader:
        current_pressure = batch["source_pressure"]
        original_pressures.append(current_pressure)
        batch_patchified = patchify_batch(current_pressure, patch_size)
        print(f'patched_pressure shape: {batch_patchified.shape}')
        #reconstructed_pressure = unpatcher(patched_pressure)
        if len(original_pressures) > max_samples_to_process:
            break

    # plot the original and reconstructed pressures
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(len(original_pressures)):
        original_image = axs[0].imshow(original_pressures[i].squeeze(0).detach().numpy(), cmap='jet', interpolation='nearest')
        unpatched_image = axs[1].imshow(reconstructed_pressures[i].squeeze(0).squeeze().detach().numpy(), cmap='jet', interpolation='nearest')
        plt.pause(0.01)
        axs[0].title.set_text('Original Pressure')
        axs[1].title.set_text('Reconstructed Pressure')
        axs[0].clear()
        axs[1].clear()
    fig.colorbar(original_image, ax=axs[0])
    fig.colorbar(unpatched_image, ax=axs[1])
    plt.show()




if __name__ == "__main__":
    main()
