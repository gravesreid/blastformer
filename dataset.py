import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, cpu_count
from utils import patchify, CFDFeatureEmbedder, custom_collate

class BlastDataset(Dataset):
    """ Dataset for blast wave simulation from BlastFoam simulator """

    def __init__(self, root_dir, k=1, normalization_file = "normalization_val.json",  normalize=True):
        self.data_dir = root_dir
        # simulation directories
        self.simulation_directories = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.k = k
        self.normalize = normalize
        self.normalization_file = normalization_file

        # make a list to store the mapping of the simulation and start timestep
        self.index_map = []

        # make a list to store the simulation json paths
        self.simulations = []

        # populate the index map

        for simulation_index, simulation_dir in enumerate(self.simulation_directories):
            json_files = sorted(
                [os.path.join(simulation_dir, f) for f in os.listdir(simulation_dir) if f.endswith('.json')],
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            self.simulations.append(json_files)
            for start_timestep in range(len(json_files) - k):
                self.index_map.append((simulation_index, start_timestep))



        # Load or compute normalization parameters
        if os.path.exists(self.normalization_file):
            self.mean, self.std = self._load_normalization()
        else:
            self.mean, self.std = self._compute_normalization()
            self._save_normalization(self.mean, self.std)

        

    def _compute_normalization(self):
        """
        Compute the normalization parameters for the dataset.
        """
        mean = 0.0
        var = 0.0
        n = 0
        simulations_processed = 0
        for sim_dir in self.simulation_directories:
            timestep_files = sorted(
                [os.path.join(sim_dir, f) for f in os.listdir(sim_dir) if f.endswith('.json')],
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            simulations_processed += 1
            print(f'Processing {simulations_processed} of {len(self.simulations)} simulations...')
            for file in timestep_files:
                with open(file, 'r') as f:
                    data = json.load(f)
                    pressures = torch.tensor(data["pressure"], dtype=torch.float32)
                    n += pressures.numel()
                    delta = pressures.mean().item() - mean
                    mean += delta * (pressures.numel() / n)
                    var += pressures.var().item() * (pressures.numel() / n)

        std = torch.sqrt(torch.tensor(var))
        return mean, std

    def _save_normalization(self, mean, std):
        """
        Save normalization parameters to a file.
        """
        std = std.item()
        with open(self.normalization_file, 'w') as f:
            json.dump({"mean": mean, "std": std}, f)
        print(f"Saved normalization parameters to {self.normalization_file}")

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
        simulation_index, start_timestep = self.index_map[idx]
        filepaths = self.simulations[simulation_index][start_timestep:start_timestep + self.k + 1]

        future_data_list = []

        for i, filepath in enumerate(filepaths):
            with open(filepath, 'r') as f:
                data = json.load(f)

                pressure_1D = torch.tensor(data["pressure"], dtype=torch.float32, requires_grad=False)
                pressure = pressure_1D.view(99,99)
                if self.normalize:
                    pressure = (pressure - self.mean) / self.std

                timestep = torch.tensor(data["time"], dtype=torch.float32, requires_grad=False).unsqueeze(0)
                wall_locations = torch.tensor(
                    [list(w.values()) for w in data["wall_locations"]],
                    dtype=torch.float32,
                    requires_grad=False
                )
                charge_data = self._process_charge_data(data["charge_data"]).detach()

                patches = patchify(pressure, 11).detach()


                future_data_list.append({
                    "pressure": patches,
                    "wall_locations": wall_locations,
                    "charge_data": charge_data,
                    "time": timestep
                })
  

        return future_data_list
 

    def _process_charge_data(self, charge_data):
        """
        Convert charge data into a flat tensor representation.
        """
        charge_tensor = torch.tensor([
            charge_data["mass"],
            *charge_data["cent0"],
            *charge_data["p10"],
        ], dtype=torch.float32, requires_grad=False)
        return charge_tensor



def main():
    root_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed_large"


    dataset = BlastDataset(root_dir, k=2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=custom_collate)

    i = 0
    for batch in dataloader:
        i += 1
        pressure = batch["pressure"]
        charge_data = batch["charge_data"]
        wall_locations = batch["wall_locations"]
        time = batch["time"]

        time_0 = time[:,0,:]
        time_1 = time[:,1,:]
        time_2 = time[:,2,:]
        print(f'time_0: {time_0.item():.10f}')
        print(f'time_1: {time_1.item():.10f}')
        print(f'time_2: {time_2.item():.10f}')
        if i == 2:
            break


if __name__ == "__main__":
    main()

