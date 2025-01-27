import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, cpu_count
from utils import patchify, CFDFeatureEmbedder

class BlastDataset(Dataset):
    """ Dataset for blast wave simulation from BlastFoam simulator """

    def __init__(self, root_dir, k=1, normalization_file = "normalization_val.json", max_timesteps=None, padding_value=0.0, normalize=True):
        self.data_dir = root_dir
        # simulation directories
        self.simulation_directories = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.k = k
        self.max_timesteps = max_timesteps
        self.padding_value = padding_value
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
        """
        Returns a tuple of length k+1 containing the data for timesteps i through i+k.
        """
        simulation_index, start_timestep = self.index_map[idx]
        filepaths = self.simulations[simulation_index][start_timestep:start_timestep+self.k+1]

        data_sequence = []

        for i, filepath in enumerate(filepaths):
            with open(filepath, 'r') as f:
                data = json.load(f)
                # extract pressure data, timestep, wall locations and charge data
                pressure = torch.tensor(data["pressure"], dtype=torch.float32)
                if self.normalize:
                    pressure = (pressure - self.mean) / self.std
                if i == 0:
                    timestep = torch.tensor(data["time"]).unsqueeze(0)
                    wall_locations = torch.tensor([list(w.values()) for w in data["wall_locations"]], dtype=torch.float32)
                    charge_data = self._process_charge_data(data["charge_data"])
                    # patchify the pressure data
                    patches = patchify(pressure, 33)
                    # embed the wall locations and charge data
                    wall_embedder = CFDFeatureEmbedder(6, 33)
                    charge_embedder = CFDFeatureEmbedder(7, 33)
                    time_embedder = CFDFeatureEmbedder(1, 33)
                    walls_embedded = []
                    for wall_loc in wall_locations:
                        wall_embedded = wall_embedder(wall_loc)
                        walls_embedded.append(wall_embedded)
                    walls_embedded = torch.stack(walls_embedded)
                    charge_embedded = charge_embedder(charge_data).unsqueeze(0)
                    time_embedded = time_embedder(timestep).unsqueeze(0)
                    data_sequence.append({
                        "pressure": patches,
                        "wall_locations": walls_embedded,
                        "charge_data": charge_embedded,
                        "time": time_embedded
                    })
                else:
                    patches = patchify(pressure, 33)
                    data_sequence.append({
                        "pressure": patches
                    })
            
        return data_sequence

    def _process_charge_data(self, charge_data):
        """
        Convert charge data into a flat tensor representation.
        """
        charge_tensor = torch.tensor([
            charge_data["mass"],
            *charge_data["cent0"],
            *charge_data["p10"],
        ], dtype=torch.float32)
        return charge_tensor

    def _pad_or_truncate(self, sequence_data):
        """
        Pad or truncate the sequence data to self.max_timesteps.
        """
        if self.max_timesteps:
            if len(sequence_data) < self.max_timesteps:
                padding = {
                    "time": 0.0,
                    "pressure": torch.full_like(sequence_data[0]["pressure"], self.padding_value),
                    "wall_locations": torch.full_like(sequence_data[0]["wall_locations"], self.padding_value),
                    "charge_data": torch.full_like(sequence_data[0]["charge_data"], self.padding_value)
                }
                sequence_data += [padding] * (self.max_timesteps - len(sequence_data))
            else:
                sequence_data = sequence_data[:self.max_timesteps]
        return sequence_data

def main():
    root_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed_large"


    dataset = BlastDataset(root_dir, max_timesteps=1069, padding_value=0.0)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=False)

    i = 0
    for batch in dataloader:
        i += 1
        print(f'batch shape: {len(batch)}')
        pressure, wall_locations, charge_data, time = batch[0]["pressure"], batch[0]["wall_locations"], batch[0]["charge_data"], batch[0]["time"]
        print(f'Pressure shape: {pressure.shape}')
        print(f'Wall locations shape: {wall_locations.shape}')
        print(f'Charge data shape: {charge_data.shape}')
        print(f'Time shape: {time.shape}')
        next_pressure = batch[1]["pressure"]
        print(f'Next pressure shape: {next_pressure.shape}')
        if i == 5:
            break


if __name__ == "__main__":
    main()

