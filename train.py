import torch
import torch.optim as optim
import torch.nn as nn
from blastformer_transformer import PressurePredictor
from dataset import *
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils import custom_collate

batch_size = 1024

# Example Usage
root_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed_large"
dataset = BlastDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate)

# Hyperparameters
input_dim = 121
hidden_dim = 256
output_dim = 121   # Predict pressures
seq_len = 302  # Number of timesteps to consider
patch_size = 11
num_layers = 4
lr = 1e-4
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model, loss, optimizer
model = PressurePredictor(input_dim, hidden_dim, num_layers, output_dim, seq_len, patch_size).to(device)
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

best_model = None
best_loss = float("inf")

epoch_losses = []

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")

            # Move inputs and targets to device
            current_pressure = batch["pressure"][:, :1, :].to(device).squeeze(1)
            next_pressures = batch["pressure"][:, 1:, :].to(device).squeeze(1)
            charge_data = batch["charge_data"][:, 0, :].unsqueeze(-1).to(device)
            wall_locations = batch["wall_locations"][:, 0, :].to(device)
            current_time = batch["time"][:, :1, :].to(device)
            next_time = batch["time"][:, 1:, :].to(device)


            # Forward pass
            outputs = model(current_pressure, charge_data, wall_locations, current_time)
            predicted_pressure = outputs[:, :next_pressures.shape[1], :]
            loss = criterion(predicted_pressure, next_pressures)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])


    # Normalize loss by the number of batches
    epoch_loss /= len(dataloader)
    epoch_losses.append(epoch_loss)

    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model = model.state_dict()

    scheduler.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Save the best model
torch.save(best_model, "pressure_predictor.pth")

# Plot the loss
plt.plot(epoch_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.show()
