import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os
import logging
import wandb

def r2_score_total(y_true_list, y_pred_list):
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    y_true_mean = np.mean(y_true)
    y_true_std = np.std(y_true)
    y_pred_mean = np.mean(y_pred)
    y_pred_std = np.std(y_pred)

    y_true = (y_true - y_true_mean) / y_true_std
    y_pred = (y_pred - y_pred_mean) / y_pred_std

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def MAPE_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    y_true: true values
    y_pred: predicted values
    """
    y_true = y_true + 1e-8  # Avoid division by zero
    mape = torch.mean(torch.abs((y_true - y_pred) / torch.abs(y_true))) 
    return mape
    
def get_iqr(data):
    """
    Calculate the interquartile range of a dataset.
    data is expected to be an unsorted list of values.
    """
    sorted_data = sorted(data)
    q1 = sorted_data[int(len(sorted_data) * 0.25)]
    q3 = sorted_data[int(len(sorted_data) * 0.75)]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [x for x in sorted_data if x < lower_bound or x > upper_bound]
    return q1, q3, iqr, lower_bound, upper_bound, outliers

def plot_iqr(q1, q3, iqr, lower_bound, upper_bound, data, title):
    """
    Plot the data and its interquartile range.
    """
    fig, ax = plt.subplots()
    data_array = np.array(data)
    print(f'data_array shape: {data_array.shape}')
    ax.boxplot(data_array)
    ax.set_title(f"Interquartile Range: {iqr:.2f}")
    ax.set_xlabel(f"Q1: {q1:.2f}, Q3: {q3:.2f}")
    ax.set_ylabel(f"Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
    ax.set_xticks([1])
    ax.set_xticklabels(["Data"])
    plt.show()

def plot_histogram(data, title, bins=10):
    """
    Plot a histogram of the data.
    """
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    plt.show()

def inverse_transform(data, min, max, normalized):
    """
    Inverse transform data that was log transformed and then min-max scaled.
    """
    if normalized:
        data = data * (max - min) + min
    data = torch.exp(data)
    return data
    
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


def visualize_results(input_pressure, target_pressure, predicted_pressure, run_name, epoch):
    """Visualizes and saves pressure field comparisons for validation."""
    num_samples = min(input_pressure.shape[0] - input_pressure.shape[0] % 9, input_pressure.shape[0])

    for i in range(0, num_samples, 3):
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for j in range(3):
            axes[j, 0].imshow(input_pressure[i+j,:,:].cpu().numpy(), cmap="jet")
            axes[j, 0].set_title(f"Input Pressure {j+i}")

            axes[j, 1].imshow(target_pressure[i+j,:,:].cpu().numpy(), cmap="jet")
            axes[j, 1].set_title(f"Target Pressure {j+i}")

            axes[j, 2].imshow(predicted_pressure[i+j,:,:].cpu().numpy(), cmap="jet")
            axes[j, 2].set_title(f"Predicted Pressure {j+i}")

        plt.tight_layout()
        vis_path = os.path.join("results", run_name, f"validation_epoch_{epoch}_sample{i}_to_{i+3}.jpg")
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        plt.savefig(vis_path)
        wandb.log({f"Validation Predictions Epoch {epoch} timestep {i} to {i+3}": wandb.Image(vis_path)})
        logging.info(f"Saved validation visualization to {vis_path}")
        plt.close(fig)

def visualize_max_pressure(true_max_pressure, predicted_max_pressure, run_name, epoch):
    """Visualizes and saves pressure field comparisons for validation."""
    num_samples = min(max(true_max_pressure.shape[0] - true_max_pressure.shape[0] % 9, true_max_pressure.shape[0]), 3)

    for i in range(0, num_samples, 3):
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for j in range(3):
            true_max = true_max_pressure[i+j,:,:].cpu().numpy()
            predicted_max = predicted_max_pressure[i+j,:,:].cpu().numpy()
            error = np.abs(true_max - predicted_max)
            im_true = axes[j, 0].imshow(true_max, cmap="jet")
            axes[j, 0].set_title(f"True Max Pressure {j+i}")
            fig.colorbar(im_true, ax=axes[j, 0], fraction=0.046, pad=0.04)
            im_pred = axes[j, 1].imshow(predicted_max, cmap="jet")
            axes[j, 1].set_title(f"Predicted Max Pressure {j+i}")
            fig.colorbar(im_pred, ax=axes[j, 1], fraction=0.046, pad=0.04)
            im_error = axes[j, 2].imshow(error, cmap="jet")
            axes[j, 2].set_title(f"Error {j+i}")
            fig.colorbar(im_error, ax=axes[j, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        vis_path = os.path.join("results", run_name, f"validation_epoch_{epoch}_sample{i}_to_{i+3}.jpg")
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        plt.savefig(vis_path)
        wandb.log({f"Validation Predictions Epoch {epoch} timestep {i} to {i+3}": wandb.Image(vis_path)})
        logging.info(f"Saved validation visualization to {vis_path}")
        plt.close(fig)

def visualize_testing(true_pressure_1, predicted_pressure_1, model_1_name, unscaled=True, vmin=None, vmax=None):
    """Show three separate figures: True Pressure, Predicted Pressure, and Error."""

    error_1 = np.abs(true_pressure_1 - predicted_pressure_1)
    error_1 = np.nan_to_num(error_1, nan=0.0, posinf=0.0, neginf=0.0)
    label = "Pressure (Pa)" if unscaled else None
    label2 = "Absolute Error (Pa)" if unscaled else None

    if vmin is not None and vmax is not None:
        vmin = vmin
        vmax = vmax

    # True Pressure
    fig_true, ax_true = plt.subplots(figsize=(6, 6))
    im_true = ax_true.imshow(true_pressure_1, cmap="jet")
    ax_true.set_title("True Pressure", fontsize=20)
    ax_true.tick_params(axis='both', which='major', labelsize=16)
    cbar_true = fig_true.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04, label=label)
    cbar_true.ax.tick_params(labelsize=16)
    plt.show()

    # Predicted Pressure
    fig_pred, ax_pred = plt.subplots(figsize=(6, 6))
    im_pred = ax_pred.imshow(predicted_pressure_1, cmap="jet")
    ax_pred.set_title("Predicted Pressure", fontsize=20)
    ax_pred.tick_params(axis='both', which='major', labelsize=16)
    cbar_pred = fig_pred.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04, label=label)
    cbar_pred.ax.tick_params(labelsize=16)
    plt.show()

    # Error
    fig_error, ax_error = plt.subplots(figsize=(6, 6))
    im_error = ax_error.imshow(error_1, cmap="binary" , vmin=vmin, vmax=vmax)
    ax_error.set_title("Error", fontsize=20)
    ax_error.tick_params(axis='both', which='major', labelsize=16)
    cbar_error = fig_error.colorbar(im_error, ax=ax_error, fraction=0.046, pad=0.04, label=label2)
    cbar_error.ax.tick_params(labelsize=16)
    plt.show()

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

def plot_recursive_predictions(sample_pressures, predicted_pressures, time_steps):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Initialize images once
    img1 = axes[0].imshow(sample_pressures[0].cpu().numpy(), cmap="jet", animated=True)
    img2 = axes[1].imshow(predicted_pressures[0].cpu().numpy(), cmap="jet", animated=True)

    axes[0].set_title("Input Pressure")
    axes[1].set_title("Predicted Pressure")

    for i, time in enumerate(time_steps):
        img1.set_data(sample_pressures[i].cpu().numpy())  # Update image data
        img2.set_data(predicted_pressures[i].cpu().numpy())

        axes[1].set_title(f"Predicted Pressure at Time: {time.item()}")
        axes[0].set_title(f"Input pressure at timestep: {i}")
        plt.pause(0.1)  # Keep minimal delay, but it’s now updating faster

    plt.show()






def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
