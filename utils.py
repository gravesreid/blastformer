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
        return self.projection(x)
    
def custom_collate(batch):
    current_batch = {key: torch.stack([item[0][key] for item in batch]) for key in batch[0][0].keys()}
    next_batch = {key: torch.stack([item[1][key] for item in batch]) for key in batch[0][1].keys()}
    return current_batch, next_batch

