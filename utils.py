import torch
import torch.nn as nn
import math

def patchify(pressure_array, patch_size):
    """
    pressure_array: torch.Tensor or np.array of shape (H, W) 
    patch_size: size of patch in each dimension (for 2D) 
    
    Return:
        patches: list of patches, each patch is flattened into a 1D array
    """
    # If 2D with shape (H, W), break it into patches of size (patch_size, patch_size)
    if len(pressure_array.shape) == 2:
        # Reshape and permute to get patches
        patches = pressure_array.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        patches = patches.contiguous().view(-1, patch_size * patch_size)
        patches = patches.float()

        return patches

def unpatchify(patches, patch_size, H, W):
    """
    patches: torch.Tensor of shape (N, patch_size*patch_size)
    patch_size: size of patch in each dimension (for 2D)
    H: height of original image
    W: width of original image
    
    Return:
        pressure_array: torch.Tensor of shape (H, W)
    """
    # Reshape patches to original image shape
    patches = patches.view(-1, H // patch_size, W // patch_size, patch_size, patch_size)
    patches = patches.permute(0, 1, 3, 2, 4).contiguous().view(H, W)
    original_array = patches
    
    return original_array

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
    
def custom_collate(batch):
    """
    Custom collation function to handle batching of multiple timesteps.
    """
    batch_size = len(batch)
    num_timesteps = len(batch[0])  # k+1 timesteps

    batched_data = {key: [] for key in batch[0][0].keys()}  # Initialize for all keys

    for sample in batch:  # Iterate over batch samples
        for t in range(num_timesteps):  # Iterate over timesteps
            for key in sample[t]:  # Iterate over data keys
                batched_data[key].append(sample[t][key])

    # Convert lists into stacked tensors
    for key in batched_data.keys():
        batched_data[key] = torch.stack(batched_data[key]).view(batch_size, num_timesteps, *batched_data[key][0].shape)

    return batched_data

