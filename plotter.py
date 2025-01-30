import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from dataset import *
from utils import unpatchify, patchify, custom_collate

def plot_simulation_sample(data_sample):
    """
    Plot the 2D pressure grid, charge location, and obstacle locations
    for all timesteps in a single simulation sample.
    
    Args:
        data_sample (dict): A sample from the dataset containing times, pressures, wall_locations, and charge_data.
    """
    times = np.array(data_sample["times"])
    print(f'times shape: {times.shape}')
    pressures = np.array(data_sample["pressures"])
    print(f'pressures shape: {pressures.shape}')
    wall_locations = data_sample["wall_locations"][0].numpy()
    charge_data = data_sample["charge_data"].numpy()

    # swap x and y axes
    pressures_flipped = np.swapaxes(pressures, 1, 2)
    
    # Extract charge centers
    cent0 = charge_data[0][1:4]  # cent0 (x, y, z)
    
    
    # Prepare figure
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        pressures_flipped[0], extent=(-4.9, 4.9, -4.9, 4.9), origin="lower", cmap="jet", alpha=0.8
    )
    cbar = fig.colorbar(im, ax=ax, label="Pressure")

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    
    # Plot walls
    for i, wall_location in enumerate(wall_locations):
        wall = patches.Rectangle(
            (wall_location[0], wall_location[1]),
            wall_location[3] - wall_location[0],
            wall_location[4] - wall_location[1],
            linewidth=1, edgecolor="r", facecolor="none", label="Wall" if i == 0 else None
        )
        ax.add_patch(wall)

    
    # Plot charge centers
    ax.plot(cent0[0], cent0[1], "o", color="blue", label="Charge Center")
    
    # Set limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.legend()
    
    # Update loop
    for t in range(len(times)):
        
        im.set_data(pressures_flipped[t])
        ax.set_title(f"Time: {t:.5f}")
        plt.pause(0.0001)  # Pause to visualize each timestep
    
    plt.show()

dataset = BlastDataset(
    root_dir="/home/reid/projects/blast_waves/dataset_parallel_processed_large"
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)
epochs = 1
i = 0
pressures = []
times = []

for batch in dataloader:
    current_pressure = batch["pressure"][:, :1, :]
    original_pressure = unpatchify(current_pressure, 11, 99, 99)
    charge_data = batch["charge_data"][:, 0, :].unsqueeze(-1)
    wall_locations = batch["wall_locations"][:, 0, :]
    current_time = batch["time"][:, :1, :]
    pressures.append(original_pressure)
    times.append(current_time)
    i += 1
    if i > 999:
        break
sample = {
    "times": times,
    "pressures": pressures,
    "wall_locations": wall_locations,
    "charge_data": charge_data
}

plot_simulation_sample(sample)

