import torch
import torch.optim as optim
import torch.nn as nn
from blastformer_transformer import BlastFormer
from hdf5_dataset import *
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils import custom_collate, scaledlp_loss, patchify_batch, unpatchify_batch, plot_reconstruction_all
import random

batch_size = 32
visualize_interval = 1
save_dir = "/home/reid/projects/blast_waves/figures/training"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


root_dir = "/home/reid/projects/blast_waves/hdf5_dataset"
train_dataset = BlastDataset(root_dir)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=min(16, os.cpu_count() - 1))



# Hyperparameters
patch_size = 33
input_dim = (99**2)//(patch_size**2)
hidden_dim = 256
output_dim = input_dim
seq_len = 302  
num_layers = 4
lr = 1e-4
epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model, loss, optimizer
model = BlastFormer(input_dim, hidden_dim, num_layers, output_dim, seq_len, patch_size).to(device)
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

best_model = None
best_loss = float("inf")

epoch_losses = []
num_epochs_since_improvement = 0

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")

            # Move inputs and targets to device
            current_pressure = batch["source_pressure"].to(device)
            next_pressures = batch["target_pressure"].to(device)
            next_patches = patchify_batch(next_pressures, patch_size)
            charge_data = batch["source_charge_data"].to(device) # shape: (batch_size, time, 7)
            wall_locations = batch["source_wall_locations"].to(device)
            current_time = batch["source_time"].to(device)
            next_time = batch["target_time"].to(device)


            # Forward pass
            outputs = model(current_pressure, charge_data, wall_locations, current_time)
            predicted_pressure = outputs[:, :next_patches.shape[1], :]
            loss = criterion(predicted_pressure, next_patches)
            scaled_loss = scaledlp_loss(predicted_pressure, next_patches, p=2, reduction="mean")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss
            tepoch.set_postfix(loss=loss.item(), scaled_loss=scaled_loss.item(), lr=optimizer.param_groups[0]['lr'])
            epoch_loss += loss.item()


    # Normalize loss by the number of batches
    epoch_loss /= len(train_dataloader)
    epoch_losses.append(epoch_loss)



   


    # Save best model
    if epoch_loss < best_loss:
        num_epochs_since_improvement = 0
        print(f"New best model found!Training loss: {epoch_loss:.6f}")
        best_loss = epoch_loss
        best_model = model.state_dict()
    else:
        num_epochs_since_improvement += 1
        print(f"No improvement in training loss. Current loss: {epoch_loss:.6f}")
        if num_epochs_since_improvement > 50:
            print("Early stopping...")
            break
    # Visualize predictions


    scheduler.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

# Save the best model
torch.save(best_model, "pressure_predictor.pth")

# Plot the loss
plt.plot(epoch_losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.show()
