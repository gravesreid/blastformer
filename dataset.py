import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, cpu_count

class BlastDataset(Dataset):
    """ Dataset for blast wave simulation from BlastFoam simulator """

    def __init__(self, root_dir, max_timesteps=None, padding_value=0.0, normalize=True):
        self.data_dir = root_dir
        self.simulations = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.max_timesteps = max_timesteps
        self.padding_value = padding_value
        self.normalize = normalize

        if normalize:
            self.pressure_mean, self.pressure_std = self._compute_normalization()


    def _compute_normalization(self):
        """
        Compute the normalization parameters for the dataset.
        """
        mean = 0.0
        var = 0.0
        n = 0
        simulations_processed = 0
        for sim_dir in self.simulations:
            timestep_files = sorted(
                [os.path.join(sim_dir, f) for f in os.listdir(sim_dir) if f.endswith('.json')],
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            simulations_processed += 1
            print(f'processing {simulations_processed} of {len(self.simulations)} simulations')
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

    def __len__(self):
        return len(self.simulations)

    def __getitem__(self, idx):
        simulation_dir = self.simulations[idx]
        timestep_files = sorted(
            [os.path.join(simulation_dir, f) for f in os.listdir(simulation_dir) if f.endswith('.json')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )

        sequence_data = []
        for file in timestep_files:
            with open(file, 'r') as f:
                data = json.load(f)
                pressure = torch.tensor(data["pressure"], dtype=torch.float32)
                if self.normalize:
                    pressure = (pressure - self.pressure_mean) / self.pressure_std
                sequence_data.append({
                    "time": data["time"],
                    "pressure": pressure,
                    "wall_locations": torch.tensor([list(w.values()) for w in data["wall_locations"]], dtype=torch.float32),
                    "charge_data": self._process_charge_data(data["charge_data"])
                })

        # Pad or truncate sequences
        sequence_data = self._pad_or_truncate(sequence_data)

        # Convert to tensors for batched processing
        times = torch.tensor([d["time"] for d in sequence_data], dtype=torch.float32)
        pressures = torch.stack([d["pressure"] for d in sequence_data])
        wall_locs = torch.stack([d["wall_locations"] for d in sequence_data])
        charge_locs = torch.stack([d["charge_data"] for d in sequence_data])

        return {
            "times": times,
            "pressures": pressures,
            "wall_locations": wall_locs,
            "charge_data": charge_locs
        }

    def _process_charge_data(self, charge_data):
        """
        Convert charge data into a flat tensor representation.
        """
        charge_tensor = torch.tensor([
            charge_data["mass"],
            *charge_data["cent0"],
            *charge_data["p10"],
            charge_data["radius0"],
            *charge_data["cent1"],
            *charge_data["p11"],
            charge_data["radius1"]
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

# Example Usage
root_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed"
dataset = BlastDataset(root_dir, max_timesteps=1069, padding_value=0.0)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

