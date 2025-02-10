import torch
import torch.optim as optim
import torch.nn as nn
from blastformer_transformer import BlastFormer
from dataset import *
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils import custom_collate, scaledlp_loss, patchify_batch, unpatchify_batch, plot_reconstruction_all
import random

batch_size = 32
visualize_interval = 1
plt.ion()
save_dir = "/home/reid/projects/blast_waves/figures/training"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


train_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed_large/train"
val_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed_large/val"
test_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed_large/test"
train_dataset = BlastDataset(train_dir)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=custom_collate)
val_dataset = BlastDataset(val_dir)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, collate_fn=custom_collate)
test_dataset = BlastDataset(test_dir)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16, collate_fn=custom_collate)


# Hyperparameters
input_dim = 81
hidden_dim = 256
output_dim = 81 
seq_len = 302  
patch_size = 11
num_layers = 4
lr = 1e-4
epochs = 20
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

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")

            # Move inputs and targets to device
            current_pressure = batch["pressure"][:, :1, :].to(device)
            next_pressures = batch["pressure"][:, -1, :].to(device)
            next_patches = patchify_batch(next_pressures, patch_size)
            charge_data = batch["charge_data"].to(device) # shape: (batch_size, time, 7)
            wall_locations = batch["wall_locations"][:, 0, :].to(device)
            current_time = batch["time"][:, :1, :].to(device)
            next_time = batch["time"][:, 1:, :].to(device)


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

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        with tqdm(val_dataloader, unit="batch") as tval:
            for batch in tval:
                current_pressure = batch["pressure"][:, :1, :].to(device).squeeze(1)
                next_pressures = batch["pressure"][:, 1:, :].to(device).squeeze(1)
                next_patches = patchify_batch(next_pressures, patch_size)
                charge_data = batch["charge_data"][:, 0, :].unsqueeze(-1).to(device)
                wall_locations = batch["wall_locations"][:, 0, :].to(device)
                current_time = batch["time"][:, :1, :].to(device)
                next_time = batch["time"][:, 1:, :].to(device)

                outputs = model(current_pressure, charge_data, wall_locations, current_time)
                predicted_pressure = outputs[:, :next_patches.shape[1], :]
                loss = criterion(predicted_pressure, next_patches)
                scaled_loss = scaledlp_loss(predicted_pressure, next_patches, p=2, reduction="mean")
                tval.set_postfix(loss=loss.item(), scaled_loss=scaled_loss.item())

                val_loss += loss.item()

        val_loss /= len(val_dataloader)


   


    # Save best model
    if val_loss < best_loss:
        print(f"New best model found! Validation loss: {val_loss:.4f}")
        best_loss = val_loss
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
