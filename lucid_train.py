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
from blastformer_transformer import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def train(args):
    setup_logging(args.run_name)
    device = args.device

    dataset = BlastDataset(args.dataset_path, split="train", normalize=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(16, os.cpu_count() - 1))
    l = len(dataloader) # used for logging

    patch_size = args.patch_size
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    seq_len = args.seq_len
    output_dim = 99
    input_dim = (99**2)//(patch_size**2)

    model = BlastFormer(input_dim, hidden_dim, num_layers, output_dim, patch_size).to(device)
    patch_embedder = PatchEmbed(1, output_dim, patch_size).to(device)
    unpatcher = UnpatchEmbed(1, output_dim, patch_size, 99**2).to(device)

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
    patience = args.patience
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_loss = 0

        for i, batch in enumerate(pbar):
            # Move inputs and targets to device
            current_pressure = batch["source_pressure"].to(device) # shape: (batch_size, 99,99)
            next_pressures = batch["target_pressure"].to(device)
            charge_data = batch["source_charge_data"].to(device) # shape: (batch_size, time, 7)
            wall_locations = batch["source_wall_locations"].to(device)
            current_time = batch["source_time"].to(device)
            next_time = batch["target_time"].to(device)

            outputs = model(current_pressure, charge_data, wall_locations, current_time)
            predicted_pressure = outputs.squeeze(1)
            loss = l1(predicted_pressure, next_pressures)


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
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=302)
    parser.add_argument('--dataset_path', type=str, default="/home/reid/projects/blast_waves/hdf5_dataset")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    launch()