import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from blastformer_transformer import BlastFormer
from hdf5_dataset import BlastDataset
from torch.utils.data import DataLoader
from utils import custom_collate, unpatchify_batch, plot_reconstruction_all
import numpy as np
import os



# Load Trained Model
patch_size = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = (99**2)//(patch_size**2)
hidden_dim = 256
output_dim = input_dim 
seq_len = 302 # Context timesteps
num_layers = 4

model = BlastFormer(input_dim, hidden_dim, num_layers, output_dim, seq_len, patch_size).to(device)
model.load_state_dict(torch.load("pressure_predictor.pth", weights_only=True))
model.eval()

# Load Dataset and Dataloader
root_dir = "/home/reid/projects/blast_waves/hdf5_dataset"
dataset = BlastDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=min(8, os.cpu_count() - 1))

true_samples = []
predicted_samples = []
times = []
charge_list = []
wall_locations_list = []
i = 0
for batch in dataloader:
    if i > 500:
        break
    print(f"Processing sample {i}")
    i += 1
    # Extract a single sample
    current_pressure = batch["source_pressure"].to(device)
    print(f'current_pressure shape: {current_pressure.shape}')
    charge_data = batch["source_charge_data"].to(device)
    charge_list.append(charge_data.detach().cpu())
    wall_locations = batch["source_wall_locations"].to(device)
    wall_locations_list.append(wall_locations.detach().cpu())
    current_time = batch["source_time"].to(device)
    next_pressures = batch["target_pressure"].to(device)
    print(f'next_pressures shape: {next_pressures.shape}')
    next_time = batch["target_time"].to(device)
    times.append(next_time.detach().cpu())




    output = model(current_pressure, charge_data, wall_locations, current_time)
    print(f'output shape: {output.shape}')
    predicted_pressures = output[:, :patch_size**2, :]


    #next_pressure_unpatched = unpatchify(next_pressures.detach().cpu().unsqueeze(1), 11, 99, 99)
    true_samples.append(next_pressures.detach().cpu())
    predicted_pressures_unpatched = unpatchify_batch(predicted_pressures.detach().cpu(), patch_size, 99, 99)
    predicted_samples.append(predicted_pressures_unpatched)


sample = {
        "times": times,
        "pressures": true_samples,
        "wall_locations": wall_locations_list[0],
        "charge_data": charge_list
    }


plot_reconstruction_all(sample, predicted_samples, index=0, show=True)
