import os
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from utils import patchify_batch, unpatchify_batch, plot_reconstruction_all
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

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
                self.mean, self.std, self.max_mean, self.max_std = self._load_normalization()
            else:
                self.mean, self.std, self.max_mean, self.max_std = self._compute_normalization()


    def _compute_normalization(self):
        """Compute mean and std of pressure values across dataset."""
        total_sum = 0.0
        total_sq_sum = 0.0
        num_elements = 0
        max_pressure_sum = 0.0
        max_pressure_sq_sum = 0.0
        max_pressure_elements = 0

        for sim_path in self.file_list:
            with h5py.File(sim_path, "r") as f:
                max_pressure = f["max_pressure_grid"][:]
                max_pressure_sum += max_pressure.sum()
                max_pressure_sq_sum += (max_pressure ** 2).sum()
                max_pressure_elements += max_pressure.size
                pressures = f["pressures"][:]
                for pressure in pressures:
                    total_sum += pressure.sum()
                    total_sq_sum += (pressure ** 2).sum()
                    num_elements += pressure.size

        mean = total_sum / num_elements
        std = ((total_sq_sum / num_elements) - (mean ** 2)) ** 0.5
        max_pressure_mean = max_pressure_sum / max_pressure_elements
        max_pressure_std = ((max_pressure_sq_sum / max_pressure_elements) - (max_pressure_mean ** 2)) ** 0.5
        print(f"Computed Normalization -> Mean: {mean:.6f}, Std: {std:.6f}")
        with open(self.normalization_file, 'w') as f:
            json.dump({"mean": float(mean), "std": float(std), "max_mean": float(max_pressure_mean), "max_std": float(max_pressure_std)}, f)
        print(f"Saved normalization parameters to {self.normalization_file}")
        return mean, std, max_pressure_mean, max_pressure_std
    
    def _load_normalization(self):
        """
        Load normalization parameters from a file.
        """
        with open(self.normalization_file, 'r') as f:
            params = json.load(f)
        print(f"Loaded normalization parameters from {self.normalization_file}")
        return params["mean"], params["std"], params["max_mean"], params["max_std"]

    def __len__(self):
    # Each file gives you (90 - 10 + 1) possible samples
        return len(self.file_list) #* (90 - 10 + 1)


    def __getitem__(self, idx):
        window_size = 30
        valid_starts = 30 - window_size + 1
        sim_idx = idx // valid_starts
        timestep_idx = idx % valid_starts
        sample_path = self.file_list[sim_idx]
        with h5py.File(sample_path, "r") as f:
            # the filename has format simulationNumber_timestepNumber.hdf5
            #extract simulation and timestep number
            filename = os.path.basename(sample_path)
            simulation_number = int(filename.split('.')[0])
            keys = list(f.keys())
            charge_center = torch.tensor(f["charge_center"], dtype=torch.float32)
            charge_mass = torch.tensor(f["charge_mass"][()].item(), dtype=torch.float32)
            wall_1 = torch.tensor(f["wall_1"], dtype=torch.float32)
            wall_2 = torch.tensor(f["wall_2"], dtype=torch.float32)
            wall_3 = torch.tensor(f["wall_3"], dtype=torch.float32)
            max_pressure = np.array(f["max_pressure_grid"], dtype=np.float32)
            max_pressure = torch.tensor(max_pressure, dtype=torch.float32)
            probe_positions = np.array(f["probe_positions"], dtype=np.float32)
            probe_positions = torch.tensor(probe_positions, dtype=torch.float32)
            input_tensor = np.array(f["input_tensor"], dtype=np.float32)
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)

            if self.normalize:
                max_pressure = (max_pressure - self.max_mean) / self.max_std

        return {
            "simulation_number": simulation_number,
            "charge_center": charge_center,
            "charge_mass": charge_mass,
            "wall_1": wall_1,
            "wall_2": wall_2,
            "wall_3": wall_3,
            "max_pressure": max_pressure,
            "probe_positions": probe_positions,
            "input_tensor": input_tensor
            }


def main():
    dataset = BlastDataset("/home/reid/projects/blast_waves/hdf5_dataset_max_pressure", normalize=True)
    dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=min(12, os.cpu_count() - 2),  # Multi-worker loading
    )

    num_processed = 0
    max_pressure_list = []
    input_tensor_list = []
    for batch in dataloader:
        if num_processed > 1000:
            break
        print(f"Processing batch {num_processed}")
        max_pressure = batch["max_pressure"]
        input_tensor = batch["input_tensor"]
        max_pressure_list.append(max_pressure)
        input_tensor_list.append(input_tensor)
        num_processed += 1


    fig, axes = plt.subplots(1, 5, figsize=(12, 4))

    print(f'max_pressure_list length: {len(max_pressure_list)}')
    for i in range(len(max_pressure_list)):
        plt.clf()  # Clear the current figure
        max_pressure = max_pressure_list[i]
        input_tensor = input_tensor_list[i]
        axes[0].imshow(max_pressure.squeeze(0), cmap="jet")
        axes[0].set_title("Max Pressure")
        axes[1].imshow(input_tensor.squeeze(0)[:,:,0], cmap="jet")
        axes[1].set_title("obstacle 1 signed distance")
        axes[2].imshow(input_tensor.squeeze(0)[:,:,1], cmap="jet")
        axes[2].set_title("obstacle 2 signed distance")
        axes[3].imshow(input_tensor.squeeze(0)[:,:,2], cmap="jet")
        axes[3].set_title("obstacle 3 signed distance")
        axes[4].imshow(input_tensor.squeeze(0)[:,:,3], cmap="jet")
        axes[4].set_title("charge signed distance")
        plt.pause(0.1)

    plt.show()

    

        

if __name__ == "__main__":
    main()