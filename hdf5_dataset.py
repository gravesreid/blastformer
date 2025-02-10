import os
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from utils import patchify_batch, unpatchify_batch, plot_reconstruction_all

class BlastDataset(Dataset):
    """Dataset for BlastFoam simulations stored in HDF5 format."""

    def __init__(self, root_dir, k=1, normalization_file = "normalization_val.json", normalize=True):
        """
        Args:
            root_dir (str): Root directory containing 'train', 'test', 'validate' HDF5 subdirectories.
            k (int): Number of timesteps per sample.
            normalize (bool): Whether to normalize the pressure data.
        """
        self.root_dir = root_dir
        self.k = k
        self.normalize = normalize
        self.normalization_file = normalization_file

        # Get all simulation files in the dataset
        self.simulation_files = []
        for split in ["train", "test", "validate"]:
            split_path = os.path.join(root_dir, split)
            if os.path.exists(split_path):
                self.simulation_files.extend([
                    os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith(".hdf5")
                ])

        # Create index mapping (simulation index, start timestep)
        self.index_map = []
        self.simulation_lengths = {}

        for sim_idx, sim_path in enumerate(self.simulation_files):
            with h5py.File(sim_path, "r") as f:
                num_timesteps = len(f.keys())  # Number of timesteps
                self.simulation_lengths[sim_idx] = num_timesteps
                for start_timestep in range(num_timesteps - k):
                    self.index_map.append((sim_idx, start_timestep))


        if os.path.exists(self.normalization_file):
            self.mean, self.std = self._load_normalization()
        else:
            self.mean, self.std = self._compute_normalization()

    def _compute_normalization(self):
        """Compute mean and std of pressure values across dataset."""
        total_sum = 0.0
        total_sq_sum = 0.0
        num_elements = 0

        for sim_path in self.simulation_files:
            with h5py.File(sim_path, "r") as f:
                for timestep in f.keys():
                    pressure = f[timestep]["pressure"][:]
                    total_sum += pressure.sum()
                    total_sq_sum += (pressure ** 2).sum()
                    num_elements += pressure.size

        mean = total_sum / num_elements
        std = ((total_sq_sum / num_elements) - (mean ** 2)) ** 0.5
        print(f"Computed Normalization -> Mean: {mean:.6f}, Std: {std:.6f}")
        return mean, std
    
    def _load_normalization(self):
        """
        Load normalization parameters from a file.
        """
        with open(self.normalization_file, 'r') as f:
            params = json.load(f)
        print(f"Loaded normalization parameters from {self.normalization_file}")
        return params["mean"], params["std"]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        sim_idx, start_timestep = self.index_map[idx]
        sim_path = self.simulation_files[sim_idx]

        future_data_list = []

        with h5py.File(sim_path, "r") as f:
            for i in range(self.k + 1):
                timestep_group = f[f"timestep_{start_timestep + i}"]
                
                pressure = torch.tensor(timestep_group["pressure"][:], dtype=torch.float32)
                if self.normalize:
                    pressure = (pressure - self.mean) / self.std

                time = torch.tensor(timestep_group["time"][:], dtype=torch.float32)
                wall_locations = torch.tensor(timestep_group["wall_locations"][:], dtype=torch.float32)
                charge_data = torch.tensor(timestep_group["charge_data"][:], dtype=torch.float32)

                future_data_list.append({
                    "pressure": pressure,
                    "wall_locations": wall_locations,
                    "charge_data": charge_data,
                    "time": time
                })

        return future_data_list


def main():
    dataset = BlastDataset("/home/reid/projects/blast_waves/hdf5_dataset", k=1, normalize=True)
    dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=min(12, os.cpu_count() - 1),  # Multi-worker loading
    pin_memory=True,  # If using GPU
    prefetch_factor=4,  # Reduce CPU bottleneck
    persistent_workers=True  # Avoid restarting workers
    )

    original_pressures = []
    reconstructed_pressures = []
    times = []
    charge_list = []

    i = 0
    for batch in dataloader:
        if i > 200:
            break
        print(f'batch length: {len(batch)}')
        pressure = batch[0]["pressure"]
        original_pressures.append(pressure)
        print(f'pressure shape: {pressure.shape}')
        charge_data = batch[0]["charge_data"]
        charge_list.append(charge_data)
        print(f'charge_data shape: {charge_data.shape}')
        wall_locations = batch[0]["wall_locations"]
        print(f'wall_locations shape: {wall_locations.shape}')
        time = batch[0]["time"]
        times.append(time)
        print(f'time shape: {time.shape}')
        print(f'time: {time}')
        next_time = batch[1]["time"]
        print(f'next time shape: {next_time.shape}')
        print(f'next time: {next_time}')
        patched_pressures = patchify_batch(pressure, 11)
        print(f'patched_pressures shape: {patched_pressures.shape}')
        unpatched_pressures = unpatchify_batch(patched_pressures, 11, 99, 99)
        print(f'unpatched_pressures shape: {unpatched_pressures.shape}')
        reconstructed_pressures.append(unpatched_pressures)

        i += 1
    sample = {
        "times": times,
        "pressures": original_pressures,
        "wall_locations": wall_locations,
        "charge_data": charge_list
    }
    plot_reconstruction_all(sample, reconstructed_pressures, index=0, show=True)
        

if __name__ == "__main__":
    main()