import os
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

class BlastDataset(Dataset):
    """Dataset for BlastFoam simulations stored in HDF5 format."""

    def __init__(self, root_dir, normalize=True, log_transform=True, split="train", show_stats=False, show_normalized_stats=False):
        """
        Args:
            root_dir (str): Root directory containing 'train', 'test', 'val' HDF5 subdirectories.
            k (int): Number of timesteps per sample.
            standardize (bool): Whether to standardize the pressure data.
        """
        self.root_dir = root_dir
        self.normalize = normalize
        self.split = split
        self.log_transform = log_transform

        # Get all simulation files in the dataset
        self.file_list = []

        split_path = os.path.join(root_dir, self.split)
        if os.path.exists(split_path):
            self.file_list.extend([
                os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith(".hdf5")
            ])
            print(f"Found {len(self.file_list)} files in {split_path}")
        # remove outliers
        self.compute_statistics(show_stats=show_stats, show_normalized_stats=show_normalized_stats)

        if normalize or log_transform:
            print("Computing normalization parameters")
            self.min_max_pressure, self.max_max_pressure, self.min_input_tensor_C1, self.max_input_tensor_C1, self.min_input_tensor_C2, self.max_input_tensor_C2, self.min_input_tensor_C3, self.max_input_tensor_C3, self.min_input_tensor_C4, self.max_input_tensor_C4 = self.compute_normalize()
            if log_transform:
                self.min_max_pressure = torch.log(self.min_max_pressure)
                self.max_max_pressure = torch.log(self.max_max_pressure)


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
        # convert to torch tensors
        min_max_pressure = torch.tensor(min_max_pressure, dtype=torch.float32)
        max_max_pressure = torch.tensor(max_max_pressure, dtype=torch.float32)
        min_input_tensor_C1 = torch.tensor(min_input_tensor_C1, dtype=torch.float32)
        max_input_tensor_C1 = torch.tensor(max_input_tensor_C1, dtype=torch.float32)
        min_input_tensor_C2 = torch.tensor(min_input_tensor_C2, dtype=torch.float32)
        max_input_tensor_C2 = torch.tensor(max_input_tensor_C2, dtype=torch.float32)
        min_input_tensor_C3 = torch.tensor(min_input_tensor_C3, dtype=torch.float32)
        max_input_tensor_C3 = torch.tensor(max_input_tensor_C3, dtype=torch.float32)
        min_input_tensor_C4 = torch.tensor(min_input_tensor_C4, dtype=torch.float32)
        max_input_tensor_C4 = torch.tensor(max_input_tensor_C4, dtype=torch.float32)
        return min_max_pressure, max_max_pressure, min_input_tensor_C1, max_input_tensor_C1, min_input_tensor_C2, max_input_tensor_C2, min_input_tensor_C3, max_input_tensor_C3, min_input_tensor_C4, max_input_tensor_C4

    def compute_statistics(self, show_stats=False, show_normalized_stats=False):
        """Compute statistics of the dataset."""
        max_pressure_max_list = []
        max_pressure_min_list = []
        max_pressure_mean_list = []
        max_pressure_std_list = []
        for sim_path in self.file_list:
            print(f"Processing {sim_path}")
            with h5py.File(sim_path, "r") as f:
                max_pressure = f["max_pressure_grid"][:]
                max_pressure_max_list.append(max(max_pressure.flatten()))
                max_pressure_min_list.append(min(max_pressure.flatten()))
                max_pressure_mean_list.append(np.mean(max_pressure.flatten()))
                max_pressure_std_list.append(np.std(max_pressure.flatten()))
        print(f"Max pressure mean: {np.mean(max_pressure_max_list)}, Max pressure std: {np.std(max_pressure_max_list)}")

        # find outliers for max pressure
        max_q1, max_q3, max_iqr, max_lower_bound, max_upper_bound, max_outliers = get_iqr(max_pressure_max_list)
        print(f"Found {len(max_outliers)} outliers in max pressure")
        min_q1, min_q3, min_iqr, min_lower_bound, min_upper_bound, min_outliers = get_iqr(max_pressure_min_list)
        print(f"Found {len(min_outliers)} outliers in min pressure")
        mean_q1, mean_q3, mean_iqr, mean_lower_bound, mean_upper_bound, mean_outliers = get_iqr(max_pressure_mean_list)
        print(f"Found {len(mean_outliers)} outliers in mean pressure")
        std_q1, std_q3, std_iqr, std_lower_bound, std_upper_bound, std_outliers = get_iqr(max_pressure_std_list)

        if show_stats:
            plot_iqr(max_q1, max_q3, max_iqr, max_lower_bound, max_upper_bound, max_pressure_max_list,  "Max pressure max")
            plot_iqr(min_q1, min_q3, min_iqr, min_lower_bound, min_upper_bound, max_pressure_min_list, "Max pressure min")
            plot_iqr(mean_q1, mean_q3, mean_iqr, mean_lower_bound, mean_upper_bound, max_pressure_mean_list, "Max pressure mean")
            plot_iqr(std_q1, std_q3, std_iqr, std_lower_bound, std_upper_bound, max_pressure_std_list, "Max pressure std")
            plot_histogram(max_pressure_max_list, "Max pressure max")
            plot_histogram(max_pressure_min_list, "Max pressure min")
            plot_histogram(max_pressure_mean_list, "Max pressure mean")
            plot_histogram(max_pressure_std_list, "Max pressure std")


        for sim_path in self.file_list:
            with h5py.File(sim_path, "r") as f:
                max_pressure = f["max_pressure_grid"][:]
                charge_mass = f["charge_mass"][()].item()
                if max(max_pressure.flatten()) in max_outliers:
                    print(f"Outlier found in {sim_path}, charge mass: {charge_mass}")
                    self.file_list.remove(sim_path)
                elif min(max_pressure.flatten()) in min_outliers:
                    print(f"Outlier found in {sim_path}, charge mass: {charge_mass}")
                    self.file_list.remove(sim_path)
                elif np.mean(max_pressure.flatten()) in mean_outliers:
                    print(f"Outlier found in {sim_path}, charge mass: {charge_mass}")
                    self.file_list.remove(sim_path)
                elif np.std(max_pressure.flatten()) in std_outliers:
                    print(f"Outlier found in {sim_path}, charge mass: {charge_mass}")
                    self.file_list.remove(sim_path)
        
        if show_normalized_stats:
            min_max_pressure, max_max_pressure, _, _, _, _, _, _, _, _ = self.compute_normalize()
            max_pressure_list = []
            max_pressure_max_list = []
            for sim_path in self.file_list:
                with h5py.File(sim_path, "r") as f:
                    max_pressure = f["max_pressure_grid"][:]
                    max_pressure = np.array(max_pressure, dtype=np.float32)
                    max_pressure = (max_pressure - min_max_pressure) / (max_max_pressure - min_max_pressure)
                    max_pressure_list.append(max_pressure)
                    max_pressure_max_list.append(max(max_pressure.flatten()))

            max_q1, max_q3, max_iqr, max_lower_bound, max_upper_bound, max_outliers = get_iqr(max_pressure_max_list)
            print(f"Found {len(max_outliers)} outliers in normalized max pressure")
            plot_iqr(max_q1, max_q3, max_iqr, max_lower_bound, max_upper_bound, max_pressure_max_list,  "Normalized Max pressure max")
            plot_histogram(max_pressure_max_list, "Normalized Max pressure max")

            # remove outliers
            max_pressure_list = [pressure for pressure in max_pressure_list if max(pressure.flatten()) not in max_outliers]
            max_pressure_max_list = [max(pressure.flatten()) for pressure in max_pressure_list]
            max_q1, max_q3, max_iqr, max_lower_bound, max_upper_bound, max_outliers = get_iqr(max_pressure_max_list)
            print(f"Found {len(max_outliers)} outliers in normalized max pressure")
            plot_iqr(max_q1, max_q3, max_iqr, max_lower_bound, max_upper_bound, max_pressure_max_list,  "Normalized Max pressure max")
            plot_histogram(max_pressure_max_list, "Normalized Max pressure max")
            




    def __len__(self):
    # Each file gives you (90 - 10 + 1) possible samples
        return len(self.file_list) #* (90 - 10 + 1)


    def __getitem__(self, idx):
        window_size = 30
        valid_starts = 30 - window_size + 1
        sim_idx = idx // valid_starts
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

            if self.log_transform:
                max_pressure = torch.log(max_pressure)
            if self.normalize:
                max_pressure = (max_pressure - self.min_max_pressure) / (self.max_max_pressure - self.min_max_pressure)

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
    dataset_log = BlastDataset("/home/reid/projects/blast_waves/hdf5_dataset_max_pressure", normalize=False, show_stats=False, show_normalized_stats=False, log_transform=True)
    dataloader_log = DataLoader(
    dataset_log,
    batch_size=1,
    shuffle=False,
    num_workers=min(12, os.cpu_count() - 2),  # Multi-worker loading
    )
    dataset_log_normal = BlastDataset("/home/reid/projects/blast_waves/hdf5_dataset_max_pressure", normalize=True, show_stats=False, show_normalized_stats=False, log_transform=True)
    dataloader_log_normal = DataLoader(
    dataset_log_normal,
    batch_size=1,
    shuffle=False,
    num_workers=min(12, os.cpu_count() - 2),  # Multi-worker loading
    )
    dataset = BlastDataset("/home/reid/projects/blast_waves/hdf5_dataset_max_pressure", normalize=True, show_stats=False, show_normalized_stats=False, log_transform=False)
    dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=min(12, os.cpu_count() - 2),  # Multi-worker loading
    )


    num_processed = 0
    max_pressure_list = []
    input_tensor_list = []
    charge_mass_list = []
    for batch in dataloader:
        print(f"Processing batch {num_processed} (dataloader)")
        max_pressure = batch["max_pressure"]
        print(f"Max pressure shape: {max_pressure.shape}")
        max_pressure = max_pressure.view(-1)
        print(f"Max pressure shape: {max_pressure.shape}")
        input_tensor = batch["input_tensor"]
        max_pressure_list.append(max_pressure)
        input_tensor_list.append(input_tensor)

        charge_mass = batch["charge_mass"]
        charge_mass = charge_mass.item()
        charge_mass_list.append(charge_mass)
        num_processed += 1

    num_processed_log = 0
    max_pressure_list_log = []
    input_tensor_list_log = []
    charge_mass_list_log = []
    for batch in dataloader_log:
        print(f"Processing batch {num_processed_log} (dataloader_log)")
        max_pressure = batch["max_pressure"]
        print(f"Max pressure shape: {max_pressure.shape}")
        max_pressure = max_pressure.view(-1)
        print(f"Max pressure shape: {max_pressure.shape}")
        input_tensor = batch["input_tensor"]
        max_pressure_list_log.append(max_pressure)
        input_tensor_list_log.append(input_tensor)

        charge_mass = batch["charge_mass"]
        charge_mass = charge_mass.item()
        charge_mass_list_log.append(charge_mass)
        num_processed_log += 1

    num_processed_log_normal = 0
    max_pressure_list_log_normal = []
    input_tensor_list_log_normal = []
    charge_mass_list_log_normal = []
    for batch in dataloader_log_normal:
        print(f"Processing batch {num_processed_log_normal} (dataloader_log_normal)")
        max_pressure = batch["max_pressure"]
        print(f"Max pressure shape: {max_pressure.shape}")
        max_pressure = max_pressure.view(-1)
        print(f"Max pressure shape: {max_pressure.shape}")
        input_tensor = batch["input_tensor"]
        max_pressure_list_log_normal.append(max_pressure)
        input_tensor_list_log_normal.append(input_tensor)

        charge_mass = batch["charge_mass"]
        charge_mass = charge_mass.item()
        charge_mass_list_log_normal.append(charge_mass)
        num_processed_log_normal += 1

    charge_mass_array = np.array(charge_mass_list)
    plt.hist(charge_mass_array, bins=10, alpha=0.7, range=(np.min(charge_mass_array), np.max(charge_mass_array)))
    plt.xlabel("Charge Mass")
    plt.ylabel("Frequency")
    plt.title("Charge Mass Histogram")
    plt.show()

    # Combine all pressures for the three datasets
    all_pressures = torch.cat(max_pressure_list)
    all_pressures_log = torch.cat(max_pressure_list_log)
    all_pressures_log_normal = torch.cat(max_pressure_list_log_normal)

    # Convert to numpy arrays for plotting
    all_pressures_np = all_pressures.numpy()
    all_pressures_log_np = all_pressures_log.numpy()
    all_pressures_log_normal_np = all_pressures_log_normal.numpy()

    # Plot histograms
    # Create histograms using subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    bins = 10

    axes[0].hist(all_pressures_np, bins=bins, alpha=0.7, range=(np.min(all_pressures_np), np.max(all_pressures_np)), color='blue')
    axes[0].set_title("Normalized", fontsize=22)
    axes[0].set_xlabel("Max Pressure", fontsize=20)
    axes[0].set_ylabel("Frequency", fontsize=20)
    axes[0].tick_params(axis='x', labelsize=16)
    axes[0].tick_params(axis='y', labelsize=16)
    axes[0].yaxis.get_offset_text().set_fontsize(16)

    axes[1].hist(all_pressures_log_np, bins=bins, alpha=0.7, range=(np.min(all_pressures_log_np), np.max(all_pressures_log_np)), color="green")
    axes[1].set_title("Log", fontsize=22)
    axes[1].set_xlabel("Max Pressure", fontsize=20)
    axes[1].tick_params(axis='x', labelsize=16)
    axes[1].tick_params(axis='y', labelsize=16)

    axes[2].hist(all_pressures_log_normal_np, bins=bins, alpha=0.7, range=(np.min(all_pressures_log_normal_np), np.max(all_pressures_log_normal_np)), color="red")
    axes[2].set_title("Log Normalized", fontsize=22)
    axes[2].set_xlabel("Max Pressure", fontsize=20)
    axes[2].tick_params(axis='x', labelsize=16)
    axes[2].tick_params(axis='y', labelsize=16)

    plt.suptitle("Max Pressure Histograms", fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Create three separate box and whisker plots using subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Combine data into a list for boxplot
    data = [all_pressures_np, all_pressures_log_np, all_pressures_log_normal_np]
    labels = ["Normalized", "Log", "Log Normalized"]

    # Create individual boxplots
    for i, ax in enumerate(axes):
        ax.boxplot([data[i]], labels=[labels[i]], showmeans=True)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        if i == 0:
            ax.set_ylabel("Max Pressure", fontsize=20)

    plt.suptitle("Max Pressure Box and Whisker Plots", fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

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