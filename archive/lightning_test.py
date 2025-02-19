import torch
from torch import utils
from archive.dataset import BlastDataset
from utils import custom_collate, unpatchify_batch, plot_reconstruction_all
from archive.blastformer_lightning import LightningBlastFormer
from blastformer_transformer import BlastFormer

test_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed_large/test"

blastformer = BlastFormer(input_dim=81, hidden_dim=256, num_layers=4, output_dim=81, seq_len=302, patch_size=11)

# Load the Lightning model with the instantiated BlastFormer model
model = LightningBlastFormer.load_from_checkpoint(
    "lightning_logs/version_3/checkpoints/epoch=19-step=86820.ckpt",
    model=blastformer  # Pass the actual instance, not the class
)

dataset = BlastDataset(test_dir)
dataloader = utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, collate_fn=custom_collate)

true_samples = []
predicted_samples = []
times = []

i = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for batch in dataloader:
    if i > 500:
        break
    print(f"Processing sample {i}")
    i += 1

    # Extract a single sample
    current_pressure = batch["pressure"][:, :1, :].to(device)
    charge_data = batch["charge_data"].to(device)
    print(f'charge_data shape: {charge_data.shape}')
    wall_locations = batch["wall_locations"].to(device)
    current_time = batch["time"][:, :1, :].to(device)
    next_pressures = batch["pressure"][:, -1, :].to(device)
    next_time = batch["time"][:, 1:, :].to(device)
    times.append(next_time.detach().cpu())

    with torch.no_grad():  # Ensure inference mode
        output = model(current_pressure, charge_data, wall_locations[:, 0, :], current_time)
        print(f'output shape: {output.shape}')
    
    predicted_pressures = output[:, :121, :]

    true_samples.append(next_pressures.detach().cpu())
    predicted_pressures_unpatched = unpatchify_batch(predicted_pressures, 11, 99, 99)
    predicted_samples.append(predicted_pressures_unpatched.detach().cpu())

sample = {
    "pressures": true_samples,
    "charge_data": batch["charge_data"],
    "wall_locations": batch["wall_locations"],
    "times": times
}


plot_reconstruction_all(sample, predicted_samples, index=0, show=True)

