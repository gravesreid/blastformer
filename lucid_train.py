import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import logging
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import math
import matplotlib.pyplot as plt
from utils import *
from hdf5_dataset import *
from lucid_blastformer import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def train(args):
    setup_logging(args.run_name)
    device = args.device

    dataset = BlastDataset(args.dataset_path, split="test", normalize=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    l = len(dataloader) # used for logging

    patch_size = args.patch_size

    model = SimpleViT(image_size=99**2, patch_size=patch_size, output_dim=99**2, dim=256, depth=4, heads=4, mlp_dim=32).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    # Wandb setup
    wandb.init(project='blastformer', name=args.run_name, config=args)
    config = wandb.config
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr

    training_loss = []

    best_loss = float('inf')
    patience = 200
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss = 0

        for i, batch in enumerate(pbar):
            # Move inputs and targets to device
            current_pressure = batch["source_pressure"].to(device)
            print(f'current_pressure shape: {current_pressure.shape}')
            current_patches = patchify_batch(current_pressure, patch_size)
            print(f'current_patches shape: {current_patches.shape}')
            next_pressures = batch["target_pressure"].to(device)
            print(f'next_pressures shape: {next_pressures.shape}')
            next_patches = patchify_batch(next_pressures, patch_size)
            print(f'next_patches shape: {next_patches.shape}')
            charge_data = batch["source_charge_data"].to(device) # shape: (batch_size, time, 7)
            print(f'charge_data shape: {charge_data.shape}')
            wall_locations = batch["source_wall_locations"].to(device)
            print(f'wall_locations shape: {wall_locations.shape}')
            current_time = batch["source_time"].to(device)
            print(f'current_time shape: {current_time.shape}')
            next_time = batch["target_time"].to(device)
            print(f'next_time shape: {next_time.shape}')

            outputs = model(current_pressure, charge_data, wall_locations, current_time)
            print(f'outputs shape: {outputs.shape}')
            predicted_patches = outputs[:, :patch_size**2, :]
            print(f'predicted_patches shape: {predicted_patches.shape}')
            loss = l1(current_patches, predicted_patches)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(epoch_loss=loss.item(), learning_rate=current_lr)
            wandb.log({
                "Batch Loss": loss.item(),
                "Learning Rate": current_lr,
                "Epoch": epoch
            })
            logger.add_scalar(f"loss: {epoch}", loss.item(), global_step=epoch * l + i)
            logger.add_scalar("learning_rate", current_lr, global_step=epoch * l + i)

        epoch_loss /= len(dataloader)
        training_loss.append(epoch_loss)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join("models", args.run_name, "best_model.pt"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping after {epoch} epochs.")
            break

        current_lr = optimizer.param_groups[0]['lr']
        logger.add_scalar("learning_rate", current_lr, global_step=epoch)
        logging.info(f"Epoch {epoch} completed. Learning rate: {current_lr}, epochs no improvement: {epochs_no_improve}, best loss: {best_loss}")


    # Save the loss plot
    plt.plot(training_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    loss_curve_path = os.path.join("results", args.run_name, "training_loss.jpg")
    plt.savefig(loss_curve_path)

    # Log the loss curve image to WandB
    wandb.log({"Training Loss Curve": wandb.Image(loss_curve_path)})
    wandb.save(loss_curve_path)


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="lucid_blastformer_0")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=3)
    parser.add_argument('--dataset_path', type=str, default="/home/reid/projects/blast_waves/hdf5_dataset")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    launch()