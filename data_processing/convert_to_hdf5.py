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
    for index in tqdm(range(total_timesteps - 1), desc=f"Processing {sim_dir}", leave=False):
        source_index = index
        target_index = index + 1
        source_json_file = json_files[source_index]
        with open(os.path.join(sim_dir, source_json_file), "r") as f:
            source_data = json.load(f)
        target_json_file = json_files[target_index]
        with open(os.path.join(sim_dir, target_json_file), "r") as f:
            target_data = json.load(f)
        source_pressure = np.array(source_data["pressure"], dtype=np.float32).reshape(99, 99)
        source_time = np.array([source_data["time"]], dtype=np.float32)
        source_wall_locations = np.array([list(w.values()) for w in source_data["wall_locations"]], dtype=np.float32)
        source_charge_data = np.array([
            source_data["charge_data"]["mass"],
            *source_data["charge_data"]["cent0"],
            *source_data["charge_data"]["p10"],
        ], dtype=np.float32)
        target_pressure = np.array(target_data["pressure"], dtype=np.float32).reshape(99, 99)
        target_time = np.array([target_data["time"]], dtype=np.float32)
        target_wall_locations = np.array([list(w.values()) for w in target_data["wall_locations"]], dtype=np.float32)
        target_charge_data = np.array([
            target_data["charge_data"]["mass"],
            *target_data["charge_data"]["cent0"],
            *target_data["charge_data"]["p10"],
        ], dtype=np.float32)

        # now make a new hdf5 file for each pair of timesteps
        with h5py.File(output_hdf5.replace(".hdf5", f"_{source_index}.hdf5"), "w") as hdf5_file:
            grp = hdf5_file.create_group(f"timestep_{source_index}")
            grp.create_dataset("source_pressure", data=source_pressure)
            grp.create_dataset("source_time", data=source_time)
            grp.create_dataset("source_wall_locations", data=source_wall_locations)
            grp.create_dataset("source_charge_data", data=source_charge_data)
            grp.create_dataset("target_pressure", data=target_pressure)
            grp.create_dataset("target_time", data=target_time)
            grp.create_dataset("target_wall_locations", data=target_wall_locations)
            grp.create_dataset("target_charge_data", data=target_charge_data)



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
    output_hdf5_dir = "/home/reid/projects/blast_waves/hdf5_dataset"  
    convert_dataset_to_hdf5(root_dataset_dir, output_hdf5_dir)
