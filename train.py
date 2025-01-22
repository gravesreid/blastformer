import torch
import torch.optim as optim
import torch.nn as nn
from blastformer_transformer import PressurePredictor
from dataset import *

# Hyperparameters
input_dim = dataset[0]["pressures"].shape[1] + dataset[0]["charge_data"].shape[1] + 18  # Combine pressure and charge features
hidden_dim = 256
output_dim = dataset[0]["pressures"].shape[1]  # Predict pressures
seq_len = 10  # Number of timesteps to consider
num_layers = 4
lr = 1e-3
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model, loss, optimizer
model = PressurePredictor(input_dim, hidden_dim, num_layers, output_dim, seq_len).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

best_model = None
best_loss = float("inf")

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        # Move inputs and targets to device
        pressures = batch["pressures"][:, :-1, :].to(device)  # Input sequence (excluding last timestep)
        targets = batch["pressures"][:, 1:, :].to(device)  # Next timestep's pressures
        charge_data = batch["charge_data"][:, :-1, :].to(device)  # Charge data
        wall_locations = batch["wall_locations"][:, :-1, :].to(device)  # Wall locations
        batch_size, seq_len, num_walls, num_wall_features = wall_locations.shape
        wall_locations_flat = wall_locations.view(batch_size, seq_len, -1) 

        # Combine features for inputs
        inputs = torch.cat([pressures, charge_data, wall_locations_flat], dim=-1)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model = model.state_dict()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

# save the best model
torch.save(best_model, "pressure_predictor.pth")
