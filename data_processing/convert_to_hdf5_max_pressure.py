import os
import json
import h5py
import numpy as np
from tqdm import tqdm  # Progress bar

def signed_distance_to_box(point, box):
    """
    Compute the signed distance from a 2D point to an axis-aligned box.
    
    Args:
        point: tuple or array (x, y)
        box: tuple or array (x_min, y_min, x_max, y_max)
        
    Returns:
        float: signed distance
            - If the point is inside the box, return the negative minimum distance to any boundary.
            - Otherwise, return the Euclidean distance to the box.
    """
    x, y = point
    x_min, y_min, x_max, y_max = box

    # Compute differences in x and y directions
    dx = max(x_min - x, 0, x - x_max)
    dy = max(y_min - y, 0, y - y_max)
    outside_distance = np.sqrt(dx*dx + dy*dy)
    
    if x_min <= x <= x_max and y_min <= y <= y_max:
        # If inside, return the negative minimum distance to a boundary.
        inside_distance = -min(x - x_min, x_max - x, y - y_min, y_max - y)
        return inside_distance
    else:
        return outside_distance
    

def signed_distance_to_point(point, center):
    """
    Compute a signed distance from a point to a charge center.
    For a point charge, you might simply use the Euclidean distance.
    Alternatively, if you wish to define an "inside" region around the charge,
    you could define a radius and return a negative value if within that radius.
    
    For simplicity, here we return the Euclidean distance.
    """
    point = np.array(point)
    center = np.array(center)
    return np.linalg.norm(point - center)

def compute_input_tensor(grid, obstacles, charge_center, charge_mass):
    """
    Compute the 4-channel input for each probe in the grid.
    
    Args:
        grid: numpy array of shape (H, W, 2) with (x,y) positions of probes.
        obstacles: list of 3 obstacles, each defined as a tuple 
                   (x_min, y_min, z_min, x_max, y_max, z_max). We'll use only x and y.
        charge_center: tuple (x, y, z) for the charge center.
        charge_mass: float, the mass of the charge.
        
    Returns:
        input_tensor: numpy array of shape (H, W, 4)
                      Channels:
                        0: Signed distance to obstacle 1
                        1: Signed distance to obstacle 2
                        2: Signed distance to obstacle 3
                        3: charge_mass * (distance from charge center)
    """
    H, W, _ = grid.shape
    input_tensor = np.zeros((H, W, 4), dtype=np.float32)
    
    # For obstacles, extract x_min, y_min, x_max, y_max (ignoring z)
    boxes = []
    for obs in obstacles:
        x_min, y_min, _, x_max, y_max, _ = obs
        boxes.append((x_min, y_min, x_max, y_max))
    
    # For each probe in the grid, compute the 4 channels
    for i in range(H):
        for j in range(W):
            p = grid[i, j]  # (x, y)
            # Channel 0-2: Signed distances to each obstacle
            d1 = signed_distance_to_box(p, boxes[0])
            d2 = signed_distance_to_box(p, boxes[1])
            d3 = signed_distance_to_box(p, boxes[2])
            # Channel 3: Charge feature (distance times mass)
            d_charge = signed_distance_to_point(p, (charge_center[0], charge_center[1]))
            channel4 = charge_mass * d_charge

            input_tensor[i, j, 0] = d1
            input_tensor[i, j, 1] = d2
            input_tensor[i, j, 2] = d3
            input_tensor[i, j, 3] = channel4
            
    return input_tensor

def generate_probe_positions():
    """
    Generate probe positions based on the original grid spacing.
    
    Returns:
        np.ndarray: Array of shape (num_probes, 3) containing (x, y, z) positions.
    """
    x = np.arange(-4.9, 5.0, 0.1)
    y = np.arange(-4.9, 5.0, 0.1)
    grid = np.array([[i, j] for i in x for j in y], dtype=np.float32).reshape(99, 99, 2)
    return grid

def convert_simulation_to_hdf5(sim_dir, output_hdf5, probe_positions):
    """
    Convert all JSON timestep files within a simulation directory into a single HDF5 file.
    
    Args:
        sim_dir (str): Path to the simulation directory.
        output_hdf5 (str): Path to the output HDF5 file.
    """
    json_files = sorted(
        [f for f in os.listdir(sim_dir) if f.endswith('.json')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])  # Sort by timestep index
    ) # this is a list of the json files for the simulation run

    # Now we break up the files into data samples. Call the source timestep_i, and the target timestep_i+1
    total_timesteps = len(json_files)

    # the wall location and charge information is constant for the entire simulation
    # so we can just load it once and use it for all samples
    with open(os.path.join(sim_dir, json_files[0]), "r") as f:
        first_data = json.load(f)
    wall_locations = np.array([list(w.values()) for w in first_data["wall_locations"]], dtype=np.float32)
    wall_1 = wall_locations[0, :]
    wall_2 = wall_locations[1, :]
    wall_3 = wall_locations[2, :]
    charge_center = np.array(first_data["charge_data"]["cent0"], dtype=np.float32)
    charge_mass = np.float32(first_data["charge_data"]["mass"])

    pressures = []  
    if total_timesteps > 900:
        for index in tqdm(range(total_timesteps), desc=f"Processing {sim_dir}", leave=False):
            # skip if total timesteps is less than 900
            # stop after 900 timesteps
            if index == 900:
                break
            json_file = json_files[index]
            with open(os.path.join(sim_dir, json_file), "r") as f:
                data = json.load(f)

            # Convert JSON data to numpy arrays
            pressure = np.array(data["pressure"], dtype=np.float32).reshape(99, 99)
            

            pressures.append(pressure)

        # convert the times and pressures to numpy arrays. Pressure is a 3D array, times is a 1D array
        all_pressures = np.array(pressures, dtype=np.float32)
        max_pressure_grid = np.max(all_pressures, axis=0)
        pressures = np.array(pressures, dtype=np.float32)

        input_tensor = compute_input_tensor(probe_positions, wall_locations, charge_center, charge_mass)

        with h5py.File(output_hdf5, "w") as hdf5_file:
            hdf5_file.create_dataset("wall_1", data=wall_1)
            hdf5_file.create_dataset("wall_2", data=wall_2)
            hdf5_file.create_dataset("wall_3", data=wall_3)
            hdf5_file.create_dataset("charge_center", data=charge_center)
            hdf5_file.create_dataset("charge_mass", data=charge_mass)


            hdf5_file.create_dataset("probe_positions", data=probe_positions, shape=(99, 99, 2))
            hdf5_file.create_dataset("max_pressure_grid", data=max_pressure_grid, shape=(99, 99))
            hdf5_file.create_dataset("input_tensor", data=input_tensor, shape=(99, 99, 4))





def convert_dataset_to_hdf5(root_dir, output_dir):
    """
    Convert the entire dataset structure (train/test/validate) into HDF5 files.

    Args:
        root_dir (str): Root directory containing train/test/validate subdirectories.
        output_dir (str): Destination directory for HDF5 files.
    """
    os.makedirs(output_dir, exist_ok=True)

    probe_positions = generate_probe_positions()

    for split in ["train", "test", "val"]:
        split_dir = os.path.join(root_dir, split)
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        if not os.path.exists(split_dir):
            print(f"⚠️ Warning: {split_dir} does not exist. Skipping...")
            continue

        for sim_dir in tqdm(os.listdir(split_dir), desc=f"Processing {split} dataset"):
            full_sim_dir = os.path.join(split_dir, sim_dir)
            if not os.path.isdir(full_sim_dir):
                continue

            output_hdf5 = os.path.join(split_output_dir, f"{sim_dir}.hdf5")
            convert_simulation_to_hdf5(full_sim_dir, output_hdf5, probe_positions)


if __name__ == "__main__":
    root_dataset_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed_2"  
    output_hdf5_dir = "/home/reid/projects/blast_waves/hdf5_dataset_max_pressure_2"  
    convert_dataset_to_hdf5(root_dataset_dir, output_hdf5_dir)
