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
    
    for index in tqdm(range(0, total_timesteps - 20, 10), desc=f"Processing {sim_dir}", leave=False):
        source_samples = []
        target_samples = []
        source_indexes = list(range(index, index + 10))  # Fixes indexing issue

        for source_index in source_indexes:
            source_json_file = json_files[source_index]
            with open(os.path.join(sim_dir, source_json_file), "r") as f:
                source_data = json.load(f)

            target_json_file = json_files[source_index + 10]
            with open(os.path.join(sim_dir, target_json_file), "r") as f:
                target_data = json.load(f)

            # Convert JSON data to numpy arrays
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

            # Append to lists
            source_samples.append({
                "pressure": source_pressure,
                "time": source_time,
                "wall_locations": source_wall_locations,
                "charge_data": source_charge_data,
            })
            target_samples.append({
                "pressure": target_pressure,
                "time": target_time,
                "wall_locations": target_wall_locations,
                "charge_data": target_charge_data,
            })

    # Save all 10 samples in a single HDF5 file
    output_hdf5_path = output_hdf5.replace(".hdf5", f"_{index}.hdf5")
    with h5py.File(output_hdf5_path, "w") as hdf5_file:
        grp = hdf5_file.create_group("data")
        
        # Iterate over 10 samples and save each field separately
        for i, (source_sample, target_sample) in enumerate(zip(source_samples, target_samples)):
            timestep_grp = grp.create_group(f"timestep_{i}")

            # Store each field separately
            timestep_grp.create_dataset("source_pressure", data=source_sample["pressure"])
            timestep_grp.create_dataset("source_time", data=source_sample["time"])
            timestep_grp.create_dataset("source_wall_locations", data=source_sample["wall_locations"])
            timestep_grp.create_dataset("source_charge_data", data=source_sample["charge_data"])
            
            timestep_grp.create_dataset("target_pressure", data=target_sample["pressure"])
            timestep_grp.create_dataset("target_time", data=target_sample["time"])
            timestep_grp.create_dataset("target_wall_locations", data=target_sample["wall_locations"])
            timestep_grp.create_dataset("target_charge_data", data=target_sample["charge_data"])




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
    output_hdf5_dir = "/home/reid/projects/blast_waves/hdf5_dataset_10_sample_per_file"  
    convert_dataset_to_hdf5(root_dataset_dir, output_hdf5_dir)
