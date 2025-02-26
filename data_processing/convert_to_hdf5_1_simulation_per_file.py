import os
import json
import h5py
import numpy as np
from tqdm import tqdm  # Progress bar

def convert_simulation_to_hdf5(sim_dir, output_hdf5):
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

    max_time = 0 
    times = []
    pressures = []  

    for index in tqdm(range(total_timesteps), desc=f"Processing {sim_dir}", leave=False):
        json_file = json_files[index]
        with open(os.path.join(sim_dir, json_file), "r") as f:
            data = json.load(f)

        # Convert JSON data to numpy arrays
        pressure = np.array(data["pressure"], dtype=np.float32).reshape(99, 99)
        time = np.float32(data["time"])
        
        max_time = max(max_time, time)

        times.append(time)
        pressures.append(pressure)

    # convert the times and pressures to numpy arrays. Pressure is a 3D array, times is a 1D array
    times = np.array(times, dtype=np.float32)
    pressures = np.array(pressures, dtype=np.float32)
    print(f'pressures shape: {pressures.shape}')
    # pressures is a 3D array of shape (total_timesteps, 99, 99)
    # times is a 1D array of shape (total_timesteps,)


    with h5py.File(output_hdf5, "w") as hdf5_file:
        hdf5_file.create_dataset("number_of_timesteps", data=total_timesteps)
        hdf5_file.create_dataset("max_time", data=max_time)
        hdf5_file.create_dataset("wall_1", data=wall_1)
        hdf5_file.create_dataset("wall_2", data=wall_2)
        hdf5_file.create_dataset("wall_3", data=wall_3)
        hdf5_file.create_dataset("charge_center", data=charge_center)
        hdf5_file.create_dataset("charge_mass", data=charge_mass)

        hdf5_file.create_dataset("times", data=times, shape=(total_timesteps,), chunks=(10,))
        hdf5_file.create_dataset("pressures", data=pressures, shape=(total_timesteps, 99, 99), chunks=(10, 99, 99))





def convert_dataset_to_hdf5(root_dir, output_dir):
    """
    Convert the entire dataset structure (train/test/validate) into HDF5 files.

    Args:
        root_dir (str): Root directory containing train/test/validate subdirectories.
        output_dir (str): Destination directory for HDF5 files.
    """
    os.makedirs(output_dir, exist_ok=True)

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
            convert_simulation_to_hdf5(full_sim_dir, output_hdf5)


if __name__ == "__main__":
    root_dataset_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed_large"  
    output_hdf5_dir = "/home/reid/projects/blast_waves/hdf5_dataset_1_simulation_per_file"  
    convert_dataset_to_hdf5(root_dataset_dir, output_hdf5_dir)
