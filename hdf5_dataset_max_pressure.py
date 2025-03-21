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

    def __init__(self, root_dir, standardize=True, normalize= False, split="train", show_stats=False):
        """
        Args:
            root_dir (str): Root directory containing 'train', 'test', 'val' HDF5 subdirectories.
            k (int): Number of timesteps per sample.
            standardize (bool): Whether to standardize the pressure data.
        """
        self.root_dir = root_dir
        self.standardize = standardize
        self.normalize = normalize
        self.split = split

        # Get all simulation files in the dataset
        self.file_list = []

        split_path = os.path.join(root_dir, self.split)
        if os.path.exists(split_path):
            self.file_list.extend([
                os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith(".hdf5")
            ])
            print(f"Found {len(self.file_list)} files in {split_path}")
        # remove outliers
        self.compute_statistics(show_stats=show_stats)

        if standardize:
            print("Computing standardization parameters")
            self.max_mean, self.max_std, self.input_tensor_C1_mean, self.input_tensor_C1_std, self.input_tensor_C2_mean, self.input_tensor_C2_std, self.input_tensor_C3_mean, self.input_tensor_C3_std, self.input_tensor_C4_mean, self.input_tensor_C4_std = self._compute_standardization()
        elif normalize:
            print("Computing normalization parameters")
            self.min_max_pressure, self.max_max_pressure, self.min_input_tensor_C1, self.max_input_tensor_C1, self.min_input_tensor_C2, self.max_input_tensor_C2, self.min_input_tensor_C3, self.max_input_tensor_C3, self.min_input_tensor_C4, self.max_input_tensor_C4 = self.compute_normalize()

    def _compute_standardization(self):
        """Compute mean and std of pressure values across dataset."""
        max_pressure_sum = 0.0
        max_pressure_sq_sum = 0.0
        max_pressure_elements = 0

        input_tensor_C1_sum = 0.0
        input_tensor_C1_sq_sum = 0.0
        input_tensor_C1_elements = 0
        input_tensor_C2_sum = 0.0
        input_tensor_C2_sq_sum = 0.0
        input_tensor_C2_elements = 0
        input_tensor_C3_sum = 0.0
        input_tensor_C3_sq_sum = 0.0
        input_tensor_C3_elements = 0
        input_tensor_C4_sum = 0.0
        input_tensor_C4_sq_sum = 0.0
        input_tensor_C4_elements = 0

        for sim_path in self.file_list:
            print(f"Processing {sim_path}")
            with h5py.File(sim_path, "r") as f:
                max_pressure = f["max_pressure_grid"][:]
                max_pressure_sum += max_pressure.sum()
                max_pressure_sq_sum += (max_pressure ** 2).sum()
                max_pressure_elements += max_pressure.size
                input_tensors = f["input_tensor"][:]
                input_tensor_C1 = input_tensors[:, :,  0]
                input_tensor_C2 = input_tensors[:, :,  1]
                input_tensor_C3 = input_tensors[:, :,  2]
                input_tensor_C4 = input_tensors[:, :,  3]
                input_tensor_C1_sum += input_tensor_C1.sum()
                input_tensor_C1_sq_sum += (input_tensor_C1 ** 2).sum()
                input_tensor_C1_elements += input_tensor_C1.size
                input_tensor_C2_sum += input_tensor_C2.sum()
                input_tensor_C2_sq_sum += (input_tensor_C2 ** 2).sum()
                input_tensor_C2_elements += input_tensor_C2.size
                input_tensor_C3_sum += input_tensor_C3.sum()
                input_tensor_C3_sq_sum += (input_tensor_C3 ** 2).sum()
                input_tensor_C3_elements += input_tensor_C3.size
                input_tensor_C4_sum += input_tensor_C4.sum()
                input_tensor_C4_sq_sum += (input_tensor_C4 ** 2).sum()
                input_tensor_C4_elements += input_tensor_C4.size

        max_pressure_mean = max_pressure_sum / max_pressure_elements
        max_pressure_std = ((max_pressure_sq_sum / max_pressure_elements) - (max_pressure_mean ** 2)) ** 0.5
        input_tensor_C1_mean = input_tensor_C1_sum / input_tensor_C1_elements
        input_tensor_C1_std = ((input_tensor_C1_sq_sum / input_tensor_C1_elements) - (input_tensor_C1_mean ** 2)) ** 0.5
        input_tensor_C2_mean = input_tensor_C2_sum / input_tensor_C2_elements
        input_tensor_C2_std = ((input_tensor_C2_sq_sum / input_tensor_C2_elements) - (input_tensor_C2_mean ** 2)) ** 0.5
        input_tensor_C3_mean = input_tensor_C3_sum / input_tensor_C3_elements
        input_tensor_C3_std = ((input_tensor_C3_sq_sum / input_tensor_C3_elements) - (input_tensor_C3_mean ** 2)) ** 0.5
        input_tensor_C4_mean = input_tensor_C4_sum / input_tensor_C4_elements
        input_tensor_C4_std = ((input_tensor_C4_sq_sum / input_tensor_C4_elements) - (input_tensor_C4_mean ** 2)) ** 0.5
        with open(self.standardization_file, 'w') as f:
            json.dump({"max_mean": float(max_pressure_mean), "max_std": float(max_pressure_std), "input_tensor_C1_mean": float(input_tensor_C1_mean), "input_tensor_C1_std": float(input_tensor_C1_std), "input_tensor_C2_mean": float(input_tensor_C2_mean), "input_tensor_C2_std": float(input_tensor_C2_std), "input_tensor_C3_mean": float(input_tensor_C3_mean), "input_tensor_C3_std": float(input_tensor_C3_std), "input_tensor_C4_mean": float(input_tensor_C4_mean), "input_tensor_C4_std": float(input_tensor_C4_std)}, f)
        print(f"Saved standardization parameters to {self.standardization_file}")
        return max_pressure_mean, max_pressure_std, input_tensor_C1_mean, input_tensor_C1_std, input_tensor_C2_mean, input_tensor_C2_std, input_tensor_C3_mean, input_tensor_C3_std, input_tensor_C4_mean, input_tensor_C4_std
    
    def compute_normalize(self):
        """Normalize the data if required."""
        min_max_pressure = float('inf')
        max_max_pressure = float('-inf')
        min_input_tensor_C1 = float('inf')
        max_input_tensor_C1 = float('-inf')
        min_input_tensor_C2 = float('inf')
        max_input_tensor_C2 = float('-inf')
        min_input_tensor_C3 = float('inf')
        max_input_tensor_C3 = float('-inf')
        min_input_tensor_C4 = float('inf')
        max_input_tensor_C4 = float('-inf')
        for sim_path in self.file_list:
            print(f"Processing {sim_path}")
            with h5py.File(sim_path, "r") as f:
                max_pressure = f["max_pressure_grid"][:]
                min_max_pressure = min(min_max_pressure, max_pressure.min())
                max_max_pressure = max(max_max_pressure, max_pressure.max())
                input_tensors = f["input_tensor"][:]
                input_tensor_C1 = input_tensors[:, :,  0]
                input_tensor_C2 = input_tensors[:, :,  1]
                input_tensor_C3 = input_tensors[:, :,  2]
                input_tensor_C4 = input_tensors[:, :,  3]
                min_input_tensor_C1 = min(min_input_tensor_C1, input_tensor_C1.min())
                max_input_tensor_C1 = max(max_input_tensor_C1, input_tensor_C1.max())
                min_input_tensor_C2 = min(min_input_tensor_C2, input_tensor_C2.min())
                max_input_tensor_C2 = max(max_input_tensor_C2, input_tensor_C2.max())
                min_input_tensor_C3 = min(min_input_tensor_C3, input_tensor_C3.min())
                max_input_tensor_C3 = max(max_input_tensor_C3, input_tensor_C3.max())
                min_input_tensor_C4 = min(min_input_tensor_C4, input_tensor_C4.min())
                max_input_tensor_C4 = max(max_input_tensor_C4, input_tensor_C4.max())
        print(f"Min max pressure: {min_max_pressure}, Max max pressure: {max_max_pressure}")
        print(f"Min input tensor C1: {min_input_tensor_C1}, Max input tensor C1: {max_input_tensor_C1}")
        print(f"Min input tensor C2: {min_input_tensor_C2}, Max input tensor C2: {max_input_tensor_C2}")
        print(f"Min input tensor C3: {min_input_tensor_C3}, Max input tensor C3: {max_input_tensor_C3}")
        print(f"Min input tensor C4: {min_input_tensor_C4}, Max input tensor C4: {max_input_tensor_C4}")
        return min_max_pressure, max_max_pressure, min_input_tensor_C1, max_input_tensor_C1, min_input_tensor_C2, max_input_tensor_C2, min_input_tensor_C3, max_input_tensor_C3, min_input_tensor_C4, max_input_tensor_C4

    def compute_statistics(self, show_stats=False):
        """Compute statistics of the dataset."""
        max_pressure_list = []
        for sim_path in self.file_list:
            print(f"Processing {sim_path}")
            with h5py.File(sim_path, "r") as f:
                max_pressure = f["max_pressure_grid"][:]
                max_pressure_list.append(max(max_pressure.flatten()))
        print(f"Max pressure mean: {np.mean(max_pressure_list)}, Max pressure std: {np.std(max_pressure_list)}")

        # find outliers
        sorted_max_pressure = sorted(max_pressure_list)
        q1 = sorted_max_pressure[int(0.25 * len(sorted_max_pressure))]
        q3 = sorted_max_pressure[int(0.75 * len(sorted_max_pressure))]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [x for x in max_pressure_list if x < lower_bound or x > upper_bound]
        print(f"Found {len(outliers)} outliers")

        for sim_path in self.file_list:
            with h5py.File(sim_path, "r") as f:
                max_pressure = f["max_pressure_grid"][:]
                charge_mass = f["charge_mass"][()].item()
                if max(max_pressure.flatten()) in outliers:
                    print(f"Outlier found in {sim_path}, charge mass: {charge_mass}")
                    self.file_list.remove(sim_path)




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

            if self.standardize:
                max_pressure = (max_pressure - self.max_mean) / self.max_std
                input_tensor[:,:,0] = (input_tensor[:,:,0] - self.input_tensor_C1_mean) / self.input_tensor_C1_std
                input_tensor[:,:,1] = (input_tensor[:,:,1] - self.input_tensor_C2_mean) / self.input_tensor_C2_std
                input_tensor[:,:,2] = (input_tensor[:,:,2] - self.input_tensor_C3_mean) / self.input_tensor_C3_std
                input_tensor[:,:,3] = (input_tensor[:,:,3] - self.input_tensor_C4_mean) / self.input_tensor_C4_std
            elif self.normalize:
                max_pressure = (max_pressure - self.min_max_pressure) / (self.max_max_pressure - self.min_max_pressure)
                input_tensor[:,:,0] = (input_tensor[:,:,0] - self.min_input_tensor_C1) / (self.max_input_tensor_C1 - self.min_input_tensor_C1)
                input_tensor[:,:,1] = (input_tensor[:,:,1] - self.min_input_tensor_C2) / (self.max_input_tensor_C2 - self.min_input_tensor_C2)
                input_tensor[:,:,2] = (input_tensor[:,:,2] - self.min_input_tensor_C3) / (self.max_input_tensor_C3 - self.min_input_tensor_C3)
                input_tensor[:,:,3] = (input_tensor[:,:,3] - self.min_input_tensor_C4) / (self.max_input_tensor_C4 - self.min_input_tensor_C4)

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
    dataset = BlastDataset("/home/reid/projects/blast_waves/hdf5_dataset_max_pressure_2", standardize=False, normalize=True, show_stats=True)
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


    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i in range(len(max_pressure_list)):
        for ax in axes:
            ax.clear()

        max_pressure = max_pressure_list[i]
        input_tensor = input_tensor_list[i]

        im0 = axes[0].imshow(max_pressure.squeeze(0), cmap="jet")
        axes[0].set_title("Max Pressure")


        for j in range(4):
            im = axes[j+1].imshow(input_tensor.squeeze(0)[:, :, j], cmap="jet")
            axes[j+1].set_title(f"Obstacle {j+1} signed distance" if j < 3 else "Charge signed distance")

        plt.pause(0.1)

    plt.show()

    

        

if __name__ == "__main__":
    main()