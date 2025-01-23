import torch
import torch.optim as optim
import torch.nn as nn
from blastformer_transformer import PressurePredictor
from dataset import *
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Example Usage
root_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed"
dataset = BlastDataset(root_dir, max_timesteps=1069, padding_value=0.0)
dataloader = DataLoader(dataset, batch_size=24, shuffle=True)

# Hyperparameters
input_dim = dataset[0]["pressures"].shape[1] + dataset[0]["charge_data"].shape[1] + 18  # Combine pressure and charge features
hidden_dim = 256
output_dim = dataset[0]["pressures"].shape[1]   # Predict pressures
seq_len = 10  # Number of timesteps to consider
num_layers = 4
lr = 1e-3
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model, loss, optimizer
model = PressurePredictor(input_dim, hidden_dim, num_layers, output_dim, seq_len).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

best_model = None
best_loss = float("inf")

epoch_losses = []

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        # Move inputs and targets to device
        pressures = batch["pressures"][:, :-1, :].to(device)
        targets = batch["pressures"][:, 1:, :].to(device)
        charge_data = batch["charge_data"][:, :-1, :].to(device)
        wall_locations = batch["wall_locations"][:, :-1, :].to(device)

        batch_size, seq_len, num_walls, num_wall_features = wall_locations.shape
        wall_locations_flat = wall_locations.view(batch_size, seq_len, -1)

        # Combine features for inputs
        inputs = torch.cat([pressures, charge_data, wall_locations_flat], dim=-1)

        target_charge_data = batch["charge_data"][:, 1:, :].to(device)
        target_wall_locations = batch["wall_locations"][:, 1:, :].to(device)
        target_wall_locations_flat = target_wall_locations.view(batch_size, seq_len, -1)
        augmented_targets = torch.cat([targets, target_charge_data, target_wall_locations_flat], dim=-1)

        # Forward pass
        outputs = model(inputs, augmented_targets)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

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
