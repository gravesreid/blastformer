import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, cpu_count

class BlastDataset(Dataset):
    """ Dataset for blast wave simulation from BlastFoam simulator """

    def __init__(self, root_dir, normalization_file = "normalization_val.json", max_timesteps=None, padding_value=0.0, normalize=True):
        self.data_dir = root_dir
        self.simulations = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.max_timesteps = max_timesteps
        self.padding_value = padding_value
        self.normalize = normalize
        self.normalization_file = normalization_file

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
        for sim_dir in self.simulations:
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
                    pressure = (pressure - self.mean) / self.std
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



