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
    )

    with h5py.File(output_hdf5, "w") as hdf5_file:
        for timestep_idx, json_file in enumerate(tqdm(json_files, desc=f"Processing {sim_dir}", leave=False)):
            json_path = os.path.join(sim_dir, json_file)

            # Read JSON data
            with open(json_path, "r") as f:
                data = json.load(f)

            # Extract structured data
            pressure = np.array(data["pressure"], dtype=np.float32).reshape(99, 99)
            time = np.array([data["time"]], dtype=np.float32)
            wall_locations = np.array([list(w.values()) for w in data["wall_locations"]], dtype=np.float32)
            charge_data = np.array([
                data["charge_data"]["mass"],
                *data["charge_data"]["cent0"],
                *data["charge_data"]["p10"],
            ], dtype=np.float32)

            # Create a group for each timestep
            grp = hdf5_file.create_group(f"timestep_{timestep_idx}")
            grp.create_dataset("pressure", data=pressure)
            grp.create_dataset("time", data=time)
            grp.create_dataset("wall_locations", data=wall_locations)
            grp.create_dataset("charge_data", data=charge_data)

    print(f"✅ Saved {sim_dir} -> {output_hdf5}")


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
