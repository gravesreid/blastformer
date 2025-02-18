import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

    
def patchify_batch(pressure_array, patch_size):
    """
    Patchifies a batched input tensor using CUDA if available.
    
    Args:
        pressure_array (torch.Tensor): Input tensor of shape (batch_size, H, W).
        patch_size (int): Size of the square patch.
    
    Returns:
        torch.Tensor: Patchified tensor of shape (batch_size, num_patches_H * num_patches_W, patch_size * patch_size).
    """
    # Ensure input is a tensor and on the same device
    if not isinstance(pressure_array, torch.Tensor):
        pressure_array = torch.tensor(pressure_array)

    # Ensure it has a batch dimension
    assert pressure_array.ndim == 3, "Input must have shape (batch_size, H, W)"

    unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
    
    # Extract patches using unfold
    patches = unfold(pressure_array.unsqueeze(1)) # add channel dimension. The output shape is (batch_size, patch_size * patch_size, num_patches)
    
    return patches.float()

def unpatchify_batch(patches, patch_size, H, W):
    """
    patches: torch.Tensor of shape (batch_size, num_patches, patch_size*patch_size)
    patch_size: size of patch in each dimension (for 2D)
    H: height of original image
    W: width of original image
    
    Return:
        pressure_array: torch.Tensor of shape (batch_size, H, W)
    """
    # Reshape patches to original image shape


    fold = nn.Fold(output_size=(H, W), kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
    patches = fold(patches).squeeze(1) # remove channel dimension
    original_array = patches
    
    return original_array

    
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

def scaledlp_loss(input: torch.Tensor, target: torch.Tensor, p: int = 2, reduction: str = "mean"):
    B = input.size(0)
    diff_norms = torch.norm(input.reshape(B, -1) - target.reshape(B, -1), p, 1)
    target_norms = torch.norm(target.reshape(B, -1), p, 1)
    val = diff_norms / target_norms
    if reduction == "mean":
        return torch.mean(val)
    elif reduction == "sum":
        return torch.sum(val)
    elif reduction == "none":
        return val
    else:
        raise NotImplementedError(reduction)


def plot_reconstruction_all(data_sample, reconstructed_pressures, index=0, save_dir=None, show=False):
    """
    Plot the ground truth and reconstructed pressure grid for all timesteps in the sample.
    If save_dir is provided, save each timestep's figure using a unique filename.
    Index is the batch sample index to plot.
    """
    times_og = np.array(data_sample["times"])
    print(f'times shape: {times_og.shape}')
    times = times_og[:,index,:]
    print(f'times shape: {times.shape}')
    pressures = np.array(data_sample["pressures"])[:, index,:,:]
    print(f'pressures shape: {pressures.shape}')
    wall_locations_og = data_sample["wall_locations"].numpy()
    print(f'wall_locations_og shape: {wall_locations_og.shape}')
    wall_locations = wall_locations_og[index,:,:]
    print(f'wall_locations shape: {wall_locations.shape}')
    charge_data_og = np.array(data_sample["charge_data"])
    print(f'charge_data_og shape: {charge_data_og.shape}')
    charge_data = charge_data_og[:,index,:]
    print(f'charge_data shape: {charge_data.shape}')
    print(f'chage_data sample: {charge_data[0]}')
    reconstructed_pressures = np.array(reconstructed_pressures)[:, index,:,:]
    print(f'reconstructed_pressures shape: {reconstructed_pressures.shape}')


    # Determine grid size (assumes square grid)
    pressures_flipped = np.swapaxes(pressures, 1, 2)
    print(f'pressures_flipped shape: {pressures_flipped.shape}')
    predicted_pressures = np.array(reconstructed_pressures)
    predicted_pressures_flipped = np.swapaxes(predicted_pressures, 1, 2)
    print(f'predicted_pressures_flipped shape: {predicted_pressures_flipped.shape}')

    # Prepare figure outside the loop so it can be updated
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    im_gt = axs[0].imshow(
        pressures[0], extent=(-4.9, 4.9, -4.9, 4.9), origin="lower", cmap="jet", alpha=0.8
    )
    axs[0].set_xlabel("X-axis")
    axs[0].set_ylabel("Y-axis")
    fig.colorbar(im_gt, ax=axs[0], label="Pressure")

    im_recon = axs[1].imshow(
        predicted_pressures[0], extent=(-4.9, 4.9, -4.9, 4.9), origin="lower", cmap="jet", alpha=0.8
    )
    axs[1].set_xlabel("X-axis")
    axs[1].set_ylabel("Y-axis")
    fig.colorbar(im_recon, ax=axs[1], label="Pressure")

    # Plot walls and charge center on both plots
    for ax in axs:
        for i, wall_location in enumerate(wall_locations):
            print(f'wall_location shape: {wall_location.shape}')
            wall = patches.Rectangle(
                (wall_location[0], wall_location[1]),
                wall_location[3] - wall_location[0],
                wall_location[4] - wall_location[1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
                label="Wall" if i == 0 else None,
            )
            ax.add_patch(wall)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.legend()

    # Update loop for all timesteps and save each figure
    for t in range(len(times)):
        # Extract charge centers
        #print(f'charge_data shape: {charge_data.shape}')
        cent0 = charge_data[0][1:4]  # cent0 (x, y, z)
        #print(f'cent0 shape: {cent0.shape}')
        axs[0].plot(cent0[0], cent0[1], "o", color="blue", label="Charge Center")
        # Update ground truth and reconstructed data
        im_gt.set_data(pressures_flipped[t])
        axs[0].set_title(f"Ground Truth at Time: {t:.5f}")

        im_recon.set_data(predicted_pressures_flipped[t])
        axs[1].set_title(f"Reconstructed at Time: {t:.5f}")
        plt.pause(0.1)

        # Save the current figure if a save directory is provided
        if save_dir:
            filename = f"{save_dir}/frame_{t}.png"
            plt.savefig(filename)
    if show:
        plt.plot()
        plt.show()
    else:
        plt.close(fig)


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
