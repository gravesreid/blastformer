import torch
import torch.optim as optim
import torch.nn as nn
from blastformer_transformer import PressurePredictor
from dataset import *
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import custom_collate

batch_size = 400

# Example Usage
root_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed_large"
dataset = BlastDataset(root_dir, max_timesteps=1069, padding_value=0.0)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=custom_collate)

# Hyperparameters
input_dim = 33
hidden_dim = 256
output_dim = 33   # Predict pressures
seq_len = 302  # Number of timesteps to consider
num_layers = 4
lr = 1e-3
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model, loss, optimizer
model = PressurePredictor(input_dim, hidden_dim, num_layers, output_dim, seq_len).to(device)
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

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
            current_batch, next_batch = batch

            # Move inputs and targets to device
            pressures = current_batch["pressure"].to(device)
            charge_data = current_batch["charge_data"].to(device)
            wall_locations = current_batch["wall_locations"].to(device)
            time = current_batch["time"].to(device)
            next_pressures = next_batch["pressure"].to(device)

            # Combine features for inputs
            inputs = torch.cat([pressures, charge_data, wall_locations, time], dim=1)
            next_pressures = torch.cat([next_pressures, charge_data, wall_locations, time], dim=1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, next_pressures)

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

    scheduler.step(epoch_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Save the best model
torch.save(best_model, "pressure_predictor.pth")

# Plot the loss
plt.plot(epoch_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.show()
