import torch
import torch.nn as nn
import math

def patchify(pressure_array, patch_size):
    """
    pressure_array: torch.Tensor or np.array of shape (H, W) or shape (N,) 
    patch_size: size of patch in each dimension (for 2D) or patch length for 1D
    
    Return:
        patches: list of patches, each patch is flattened into a 1D array
    """
    # If 1D with length = 10000, you might do something like:
    # break it into blocks of size patch_size
    if len(pressure_array.shape) == 1:
        num_patches = math.ceil(pressure_array.shape[0] / patch_size)
        patches = []
        for p in range(num_patches):
            start = p * patch_size
            end = start + patch_size
            patch = pressure_array[start:end]
            patches.append(patch)
        # patches is a list of arrays/tensors
        # Convert to a single tensor: shape [num_patches, patch_size]
        #patches = torch.stack([torch.tensor(p) for p in patches])
        patches = torch.stack([torch.as_tensor(p, dtype=torch.float32) for p in patches])

        return patches


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

