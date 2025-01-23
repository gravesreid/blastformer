import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from blastformer_transformer import PressurePredictor
from dataset import BlastDataset
from torch.utils.data import DataLoader

def plot_reconstruction_all(data_sample, reconstructed_pressures):
    """
    Plot the ground truth and reconstructed pressure grid for all timesteps in the sample.
    """
    times = data_sample["times"].numpy()
    pressures = data_sample["pressures"].numpy()
    wall_locations = data_sample["wall_locations"][0].numpy()
    charge_data = data_sample["charge_data"].numpy()

    # Extract charge centers
    cent0 = charge_data[0][1:4]  # cent0 (x, y, z)

    # Determine grid size (assumes square grid)
    grid_size = int((pressures.shape[1]) ** 0.5)
    assert grid_size ** 2 == pressures.shape[1], "Pressure grid is not square."

    # Prepare figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    im_gt = axs[0].imshow(
        pressures[0].reshape(grid_size, grid_size).T,
        extent=(-4, 4, -4, 4),
        origin="lower",
        cmap="jet",
        alpha=0.8,
    )
    axs[0].set_title("Ground Truth")
    axs[0].set_xlabel("X-axis")
    axs[0].set_ylabel("Y-axis")
    fig.colorbar(im_gt, ax=axs[0], label="Pressure")

    im_recon = axs[1].imshow(
        reconstructed_pressures[0].reshape(grid_size, grid_size).T,
        extent=(-4, 4, -4, 4),
        origin="lower",
        cmap="jet",
        alpha=0.8,
    )
    axs[1].set_title("Reconstructed")
    axs[1].set_xlabel("X-axis")
    axs[1].set_ylabel("Y-axis")
    fig.colorbar(im_recon, ax=axs[1], label="Pressure")

    # Plot walls and charge center on both plots
    for ax in axs:
        for i, wall_location in enumerate(wall_locations):
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
        ax.plot(cent0[0], cent0[1], "o", color="blue", label="Charge Center")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.legend()

    # Update loop for all timesteps
    for t in range(len(times)):
        # Update ground truth
        pressure_grid_gt = pressures[t].reshape(grid_size, grid_size).T
        im_gt.set_data(pressure_grid_gt)
        axs[0].set_title(f"Ground Truth at Time: {times[t]:.5f}")

        # Update reconstruction
        pressure_grid_recon = reconstructed_pressures[t].reshape(grid_size, grid_size).T
        im_recon.set_data(pressure_grid_recon)
        axs[1].set_title(f"Reconstructed at Time: {times[t]:.5f}")

        plt.pause(0.1)  # Pause to visualize each timestep

    plt.show()


# Load Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 6594  # Match your dataset (pressures + charge_data + wall_locations)
hidden_dim = 256
output_dim = 6561  # Number of pressures predicted
seq_len = 100  # Context timesteps
num_layers = 4

model = PressurePredictor(input_dim, hidden_dim, num_layers, output_dim, seq_len).to(device)
model.load_state_dict(torch.load("pressure_predictor.pth"))
model.eval()

# Load Dataset and Dataloader
root_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed"
dataset = BlastDataset(root_dir, max_timesteps=1069, padding_value=0.0)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for batch in dataloader:
    # Extract a single sample
    sample = {
        "times": batch["times"][0],
        "pressures": batch["pressures"][0],
        "wall_locations": batch["wall_locations"][0],
        "charge_data": batch["charge_data"][0],
    }

    # Initialize inputs for autoregressive prediction
    input_pressures = batch["pressures"][:, :seq_len, :].to(device)
    input_charge_data = batch["charge_data"][:, :seq_len, :].to(device)
    input_wall_locations = batch["wall_locations"][:, :seq_len, :, :].to(device)
    batch_size, seq_len, num_walls, num_wall_features = input_wall_locations.shape
    input_wall_locations_flat = input_wall_locations.view(batch_size, seq_len, -1)

    # Combine initial features for the encoder input
    src = torch.cat([input_pressures, input_charge_data, input_wall_locations_flat], dim=-1)
    print(f'Encoder input (src) shape: {src.shape}')

    # Initialize the target sequence (tgt) with the first timestep as zeros
    tgt = torch.zeros((batch_size, 1, src.shape[-1]), device=device)

    reconstructed_pressures = []
    with torch.no_grad():
        for t in range(seq_len, sample["times"].shape[0]):
            # Predict next timestep pressures
            output = model(src, tgt).cpu()  # Predict pressures for the next timestep
            next_pressure = output[:, -1, :]  # Extract the last predicted timestep
            reconstructed_pressures.append(next_pressure)

            # Log outputs for debugging
            print(f"Step {t}, Output Mean: {output.mean().item()}, Std: {output.std().item()}")

            # Prepare inputs for the next timestep
            next_input_pressures = next_pressure.unsqueeze(1).to(device)
            next_charge_data = batch["charge_data"][:, t:t + 1, :].to(device)
            next_wall_locations = batch["wall_locations"][:, t:t + 1, :, :].to(device)
            next_wall_locations_flat = next_wall_locations.view(batch_size, 1, -1)

            # Append the predicted timestep to the target sequence
            tgt = torch.cat([next_input_pressures, next_charge_data, next_wall_locations_flat], dim=-1)
            print(f'Decoder input (tgt) shape: {tgt.shape}')

            # Update the encoder input features
            next_input_features = torch.cat(
                [next_input_pressures, next_charge_data, next_wall_locations_flat], dim=-1
            )
            src = torch.cat([src[:, 1:, :], next_input_features], dim=1)

    # Stack reconstructed pressures
    reconstructed_pressures = torch.stack(reconstructed_pressures, dim=1)  # Shape: (batch_size, time_steps, output_dim)

    # Visualize the reconstruction for all timesteps
    plot_reconstruction_all(sample, reconstructed_pressures[0])
    break
