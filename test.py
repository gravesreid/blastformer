import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from blastformer_transformer import BlastFormer
from dataset import BlastDataset
from torch.utils.data import DataLoader
from utils import custom_collate, unpatchify_batch, plot_reconstruction_all
import numpy as np



# Load Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 121  # Match your dataset (pressures + charge_data + wall_locations)
hidden_dim = 256
output_dim = 121  # Number of pressures predicted
seq_len = 302 # Context timesteps
patch_size = 9
num_layers = 4

model = BlastFormer(input_dim, hidden_dim, num_layers, output_dim, seq_len, patch_size).to(device)
model.load_state_dict(torch.load("pressure_predictor.pth", weights_only=True))
model.eval()

# Load Dataset and Dataloader
root_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed_large/test"
dataset = BlastDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, collate_fn=custom_collate)

true_samples = []
predicted_samples = []
times = []
charge_list = []
i = 0
for batch in dataloader:
    if i > 500:
        break
    print(f"Processing sample {i}")
    i += 1
    # Extract a single sample
    current_pressure = batch["pressure"][:, :1, :].to(device).squeeze(1)
    print(f'current_pressure shape: {current_pressure.shape}')
    charge_data = batch["charge_data"][:, 0, :].unsqueeze(-1).to(device)
    charge_list.append(batch["charge_data"].detach().cpu())
    wall_locations = batch["wall_locations"].to(device)
    current_time = batch["time"][:, :1, :].to(device)
    next_pressures = batch["pressure"][:, 1:, :].to(device).squeeze(1)
    print(f'next_pressures shape: {next_pressures.shape}')
    next_time = batch["time"][:, 1:, :].to(device)
    times.append(next_time.detach().cpu())




    output = model(current_pressure, charge_data, wall_locations[:, 0, :], current_time)
    print(f'output shape: {output.shape}')
    predicted_pressures = output[:, :81, :]


    #next_pressure_unpatched = unpatchify(next_pressures.detach().cpu().unsqueeze(1), 11, 99, 99)
    true_samples.append(next_pressures.detach().cpu())
    predicted_pressures_unpatched = unpatchify_batch(predicted_pressures.detach().cpu(), 9, 99, 99)
    predicted_samples.append(predicted_pressures_unpatched)


    sample = {
        "times": times,
        "pressures": true_samples,
        "wall_locations": wall_locations.detach().cpu(),
        "charge_data": charge_list
    }


plot_reconstruction_all(sample, predicted_samples, index=0, show=True)
