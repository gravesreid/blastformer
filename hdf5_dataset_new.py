import os
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from utils import patchify_batch, unpatchify_batch, plot_reconstruction_all
import numpy as np

class BlastDataset(Dataset):
    """Dataset for BlastFoam simulations stored in HDF5 format."""

    def __init__(self, root_dir, normalization_file = "normalization_val.json", normalize=True, split="train"):
        """
        Args:
            root_dir (str): Root directory containing 'train', 'test', 'val' HDF5 subdirectories.
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

        if normalize:
            if os.path.exists(self.normalization_file):
                self.mean, self.std = self._load_normalization()
            else:
                self.mean, self.std = self._compute_normalization()


    def _compute_normalization(self):
        """Compute mean and std of pressure values across dataset."""
        total_sum = 0.0
        total_sq_sum = 0.0
        num_elements = 0

        for sim_path in self.file_list:
            with h5py.File(sim_path, "r") as f:
                for timestep in f.keys():
                    pressure = f[timestep]["source_pressure"][:]
                    total_sum += pressure.sum()
                    total_sq_sum += (pressure ** 2).sum()
                    num_elements += pressure.size

        mean = total_sum / num_elements
        std = ((total_sq_sum / num_elements) - (mean ** 2)) ** 0.5
        print(f"Computed Normalization -> Mean: {mean:.6f}, Std: {std:.6f}")
        with open(self.normalization_file, 'w') as f:
            json.dump({"mean": float(mean), "std": float(std)}, f)
        print(f"Saved normalization parameters to {self.normalization_file}")
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
        sim_idx = idx // 900
        timestep_idx = idx % 900
        sample_path = self.file_list[sim_idx]
        with h5py.File(sample_path, "r") as f:
            # the filename has format simulationNumber_timestepNumber.hdf5
            #extract simulation and timestep number
            filename = os.path.basename(sample_path)
            keys = list(f.keys())
            charge_center = torch.tensor(f["charge_center"], dtype=torch.float32)
            charge_mass = torch.tensor(f["charge_mass"][()].item(), dtype=torch.float32)
            wall_1 = torch.tensor(f["wall_1"], dtype=torch.float32)
            wall_2 = torch.tensor(f["wall_2"], dtype=torch.float32)
            wall_3 = torch.tensor(f["wall_3"], dtype=torch.float32)
            number_of_timesteps = torch.tensor(f["number_of_timesteps"][()].item(), dtype=torch.int32)
            max_time = torch.tensor(f["max_time"][()].item(), dtype=torch.float32)

            # fetch consecutive timesteps
            end_idx = min(timestep_idx + 10, number_of_timesteps)
            times = torch.tensor(f["times"][timestep_idx:end_idx], dtype=torch.float32)
            pressures = np.array(f["pressures"][timestep_idx:end_idx], dtype=np.float32)
            pressures = torch.tensor(pressures, dtype=torch.float32)

            # handle cases where the number of timesteps is less than 10
            if times.shape[0] < 10:
                padding = torch.zeros((10 - times.shape[0], *times.shape[1:]), dtype=torch.float32)
                times = torch.cat([times, padding], dim=0)
                pressures = torch.cat([pressures, padding.expand(10 - pressures.shape[0], 99, 99)], dim=0)

        return {
            "charge_center": charge_center,
            "charge_mass": charge_mass,
            "wall_1": wall_1,
            "wall_2": wall_2,
            "wall_3": wall_3,
            "number_of_timesteps": number_of_timesteps,
            "max_time": max_time,
            "times": times,
            "pressures": pressures,}


def main():
    dataset = BlastDataset("/home/reid/projects/blast_waves/hdf5_dataset_1_simulation_per_file", normalize=True)
    dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=min(12, os.cpu_count() - 2),  # Multi-worker loading
    )


    for batch in dataloader:
        charge_center = batch["charge_center"]
        charge_mass = batch["charge_mass"]
        wall_1 = batch["wall_1"]
        wall_2 = batch["wall_2"]
        wall_3 = batch["wall_3"]
        times = batch["times"]
        pressures = batch["pressures"]
        number_of_timesteps = batch["number_of_timesteps"]
        max_time = batch["max_time"]
        print(f"number_of_timesteps: {number_of_timesteps}")
        print(f"max_time: {max_time}")
        print(f'charge_center: {charge_center}')
        print(f'charge_mass: {charge_mass}')
        print(f'wall_1 shape: {wall_1.shape}')
        print(f'wall_2 shape: {wall_2.shape}')
        print(f'wall_3 shape: {wall_3.shape}')
        print(f'times shape: {times.shape}')
        print(f'pressures shape: {pressures.shape}')
        print(f'times: {times}')
        
        break

    

        

if __name__ == "__main__":
    main()