import os
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from utils import patchify_batch, unpatchify_batch, plot_reconstruction_all

class BlastDataset(Dataset):
    """Dataset for BlastFoam simulations stored in HDF5 format."""

    def __init__(self, root_dir, normalization_file = "normalization_val.json", normalize=True, split="train"):
        """
        Args:
            root_dir (str): Root directory containing 'train', 'test', 'validate' HDF5 subdirectories.
            k (int): Number of timesteps per sample.
            normalize (bool): Whether to normalize the pressure data.
        """
        self.root_dir = root_dir
        self.normalize = normalize
        self.normalization_file = normalization_file
        self.split = split

        # Get all simulation files in the dataset
        self.file_list = []
        split_path = os.path.join(root_dir, self.split)
        if os.path.exists(split_path):
            self.file_list.extend([
                os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith(".hdf5")
            ])

        self.file_list.sort(key=self._extract_simulation_and_timestep)


        if os.path.exists(self.normalization_file):
            self.mean, self.std = self._load_normalization()
        else:
            self.mean, self.std = self._compute_normalization()

    def _extract_simulation_and_timestep(self, filename):
        """Extracts the simulation and timestep number from the filename for sorting."""
        base_name = os.path.basename(filename)
        parts = base_name.split('_')
        simulation_number = int(parts[0])  # First part is the simulation number
        timestep_number = int(parts[1].split('.')[0])  # Second part is the timestep number
        return simulation_number, timestep_number

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
        return len(self.file_list)

    def __getitem__(self, idx):
        sample_path = self.file_list[idx]
        with h5py.File(sample_path, "r") as f:
            # the filename has format simulationNumber_timestepNumber.hdf5
            #extract simulation and timestep number
            filename = os.path.basename(sample_path)
            simulation_number, timestep_number = self._extract_simulation_and_timestep(filename)
            timestep_key = list(f.keys())[0]
            data = f[timestep_key]
            source_pressure = torch.tensor(data["source_pressure"][:], dtype=torch.float32)
            target_pressure = torch.tensor(data["target_pressure"][:], dtype=torch.float32)
            source_time = torch.tensor(data["source_time"][:], dtype=torch.float32)
            target_time = torch.tensor(data["target_time"][:], dtype=torch.float32)
            source_wall_locations = torch.tensor(data["source_wall_locations"][:], dtype=torch.float32)
            target_wall_locations = torch.tensor(data["target_wall_locations"][:], dtype=torch.float32)
            source_charge_data = torch.tensor(data["source_charge_data"][:], dtype=torch.float32)
            target_charge_data = torch.tensor(data["target_charge_data"][:], dtype=torch.float32)

            if self.normalize:
                source_pressure = (source_pressure - self.mean) / self.std
                target_pressure = (target_pressure - self.mean) / self.std

        return {
            "simulation_number": simulation_number,
            "timestep_number": timestep_number,
            "source_pressure": source_pressure,
            "target_pressure": target_pressure,
            "source_time": source_time,
            "target_time": target_time,
            "source_wall_locations": source_wall_locations,
            "target_wall_locations": target_wall_locations,
            "source_charge_data": source_charge_data,
            "target_charge_data": target_charge_data
        }


def main():
    dataset = BlastDataset("/home/reid/projects/blast_waves/hdf5_dataset", normalize=True)
    dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=min(12, os.cpu_count() - 1),  # Multi-worker loading
    )

    original_pressures = []
    reconstructed_pressures = []
    times = []
    charge_list = []

    for batch in dataloader:
        simulation_number = batch["simulation_number"]
        if simulation_number[0] != 0:
            break
        timestep_number = batch["timestep_number"]
        source_pressure = batch["source_pressure"]
        print(f'source_pressure shape: {source_pressure.shape}')
        original_pressures.append(source_pressure)
        target_pressure = batch["target_pressure"]
        source_time = batch["source_time"]
        print(f'source_time shape: {source_time.shape}')
        times.append(source_time)
        target_time = batch["target_time"]
        source_wall_locations = batch["source_wall_locations"]
        print(f'source_wall_locations shape: {source_wall_locations.shape}')
        target_wall_locations = batch["target_wall_locations"]
        source_charge_data = batch["source_charge_data"]
        print(f'source_charge_data shape: {source_charge_data.shape}')
        charge_list.append(source_charge_data)
        target_charge_data = batch["target_charge_data"]

        # Patchify the batch
        patched_source_pressure = patchify_batch(source_pressure, 11)
        print(f'patched_source_pressure shape: {patched_source_pressure.shape}')
        unpatched_source_pressure = unpatchify_batch(patched_source_pressure, 11, 99, 99)
        reconstructed_pressures.append(unpatched_source_pressure)
    
    sample = {
        "times": times,
        "pressures": original_pressures,
        "wall_locations": source_wall_locations,
        "charge_data": charge_list
    }
    plot_reconstruction_all(sample, reconstructed_pressures, index=0, save_dir=None, show=True)

        

if __name__ == "__main__":
    main()