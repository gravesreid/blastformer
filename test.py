import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from blastformer_transformer import PressurePredictor
from dataset import BlastDataset
from torch.utils.data import DataLoader
from utils import custom_collate, unpatchify
import numpy as np

def plot_reconstruction_all(data_sample, reconstructed_pressures):
    """
    Plot the ground truth and reconstructed pressure grid for all timesteps in the sample.
    """
    times = np.array(data_sample["times"])
    pressures = np.array(data_sample["pressures"])
    wall_locations = data_sample["wall_locations"][0].numpy()
    charge_data = data_sample["charge_data"].numpy()

    # Extract charge centers
    cent0 = charge_data[0][1:4]  # cent0 (x, y, z)

    # Determine grid size (assumes square grid)
    pressures_flipped = np.swapaxes(pressures, 1, 2)
    predicted_pressures = np.array(reconstructed_pressures)
    predicted_pressures_flipped = np.swapaxes(predicted_pressures, 1, 2)


    # Prepare figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    im_gt = axs[0].imshow(
        pressures_flipped[0], extent=(-4.9, 4.9, -4.9, 4.9), origin="lower", cmap="jet", alpha=0.8
    )
    axs[0].set_title("Ground Truth")
    axs[0].set_xlabel("X-axis")
    axs[0].set_ylabel("Y-axis")
    fig.colorbar(im_gt, ax=axs[0], label="Pressure")

    im_recon = axs[1].imshow(
        predicted_pressures_flipped[0], extent=(-4.9, 4.9, -4.9, 4.9), origin="lower", cmap="jet", alpha=0.8
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
        im_gt.set_data(pressures_flipped[t])
        axs[0].set_title(f"Ground Truth at Time: {t:.5f}")

        # Update reconstruction
        im_recon.set_data(predicted_pressures_flipped[t])
        axs[1].set_title(f"Reconstructed at Time: {t:.5f}")

        plt.pause(0.1)  # Pause to visualize each timestep

    plt.show()


# Load Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 121  # Match your dataset (pressures + charge_data + wall_locations)
hidden_dim = 256
output_dim = 121  # Number of pressures predicted
seq_len = 302 # Context timesteps
patch_size = 11
num_layers = 4

model = PressurePredictor(input_dim, hidden_dim, num_layers, output_dim, seq_len, patch_size).to(device)
model.load_state_dict(torch.load("pressure_predictor.pth", weights_only=True))
model.eval()

# Load Dataset and Dataloader
root_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed_large"
dataset = BlastDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)

true_samples = []
predicted_samples = []
times = []
i = 0
for batch in dataloader:
    if i > 900:
        break
    print(f"Processing sample {i}")
    i += 1
    # Extract a single sample
    current_pressure = batch["pressure"][:, :1, :].to(device).squeeze(1)
    charge_data = batch["charge_data"][:, 0, :].unsqueeze(-1).to(device)
    wall_locations = batch["wall_locations"][:, 0, :].to(device)
    current_time = batch["time"][:, :1, :].to(device)
    next_pressures = batch["pressure"][:, 1:, :].to(device).squeeze(1)
    next_time = batch["time"][:, 1:, :].to(device)
    times.append(next_time.detach().cpu())




    output = model(current_pressure, charge_data, wall_locations, current_time)
    predicted_pressures = output[:, :next_pressures.shape[1], :]


    next_pressure_unpatched = unpatchify(next_pressures.detach().cpu().unsqueeze(1), 11, 99, 99)
    true_samples.append(next_pressure_unpatched)
    predicted_pressures_unpatched = unpatchify(predicted_pressures.detach().cpu().unsqueeze(1), 11, 99, 99)
    predicted_samples.append(predicted_pressures_unpatched)


    sample = {
        "times": times,
        "pressures": true_samples,
        "wall_locations": wall_locations.detach().cpu(),
        "charge_data": charge_data.detach().cpu()
    }


plot_reconstruction_all(sample, predicted_samples)
