import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from dataset import *

def plot_simulation_sample(data_sample):
    """
    Plot the 2D pressure grid, charge location, and obstacle locations
    for all timesteps in a single simulation sample.
    
    Args:
        data_sample (dict): A sample from the dataset containing times, pressures, wall_locations, and charge_data.
    """
    times = data_sample["times"].numpy()
    pressures = data_sample["pressures"].numpy()
    wall_locations = data_sample["wall_locations"][0].numpy()
    charge_data = data_sample["charge_data"].numpy()
    
    # Extract charge centers
    cent0 = charge_data[0][1:4]  # cent0 (x, y, z)
    
    # Determine grid size (assumes square grid)
    grid_size = int(np.sqrt(pressures.shape[1]))
    assert grid_size ** 2 == pressures.shape[1], "Pressure grid is not square."
    
    # Prepare figure
    fig, ax = plt.subplots(figsize=(8, 8))
    pressure_grid = pressures[0].reshape(grid_size, grid_size).T  # Transpose the grid for correct orientation
    im = ax.imshow(
        pressure_grid, extent=(-4.9, 4.9, -4.9, 4.9), origin="lower", cmap="jet", alpha=0.8
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
        pressure_grid = pressures[t].reshape(grid_size, grid_size).T  # Update pressure grid
        im.set_data(pressure_grid)
        ax.set_title(f"Time: {times[t]:.5f}")
        plt.pause(0.0001)  # Pause to visualize each timestep
    
    plt.show()

dataset = BlastDataset(
    root_dir="/home/reid/projects/blast_waves/dataset_parallel_processed_large"
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# Example Usage
for batch in dataloader:
    sample = {
        "times": batch["times"][0],
        "pressures": batch["pressures"][0],
        "wall_locations": batch["wall_locations"][0],
        "charge_data": batch["charge_data"][0]
    }
    plot_simulation_sample(sample)
    break

