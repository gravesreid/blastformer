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

    training_dataset = BlastDataset(args.dataset_path, split="train", normalize=True)
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(16, os.cpu_count() - 1))
    l = len(training_dataloader) # used for logging

    validation_dataset = BlastDataset(args.dataset_path, split="val", normalize=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(16, os.cpu_count() - 1))
    if len(validation_dataloader) == 0:
        logging.error("Validation dataloader is empty. Check the dataset path.")
        return

    patch_size = args.patch_size
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    seq_len = args.seq_len
    output_dim = 99
    input_dim = (99**2)//(patch_size**2)

    model = BlastFormer(input_dim, hidden_dim, num_layers, output_dim, patch_size).to(device)

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
    validation_loss = []

    best_loss = float('inf')
    patience = args.patience
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(training_dataloader)
        epoch_train_loss = 0

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

            epoch_train_loss += loss.item()

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(epoch_loss=loss.item(), learning_rate=current_lr)
            wandb.log({
                "Batch Loss": loss.item(),
                "Learning Rate": current_lr,
                "Epoch": epoch
            })
            logger.add_scalar(f"loss: {epoch}", loss.item(), global_step=epoch * l + i)
            logger.add_scalar("learning_rate", current_lr, global_step=epoch * l + i)

        epoch_train_loss /= len(training_dataloader)
        training_loss.append(epoch_train_loss)
        scheduler.step(epoch_train_loss)

        # Validation
        model.eval()
        eval_model_loss = 0
        # store the first batch for visualization
        vis_inputs, vis_targets, vis_predictions = None, None, None
        with torch.no_grad():
            for j, val_batch in enumerate(validation_dataloader):
                val_current_pressure = val_batch["source_pressure"].to(device)
                val_next_pressures = val_batch["target_pressure"].to(device)
                val_charge_data = val_batch["source_charge_data"].to(device)
                val_wall_locations = val_batch["source_wall_locations"].to(device)
                val_current_time = val_batch["source_time"].to(device)

                val_outputs = model(val_current_pressure, val_charge_data, val_wall_locations, val_current_time)
                val_predicted_pressure = val_outputs.squeeze(1)
                val_loss = l1(val_predicted_pressure, val_next_pressures)
                eval_model_loss += val_loss.item()

                if j == 0:
                    vis_inputs = val_current_pressure
                    vis_targets = val_next_pressures
                    vis_predictions = val_predicted_pressure

            epoch_val_loss = eval_model_loss / len(validation_dataloader)
            validation_loss.append(epoch_val_loss)
            wandb.log({
                "Validation Loss": epoch_val_loss,
                "Epoch": epoch
            })
            logger.add_scalar("validation_loss", epoch_val_loss, global_step=epoch)
            logger.info(f"Epoch {epoch} - Training Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}")

        # visualize validation predictions
        if vis_inputs is not None:
            visualize_results(vis_inputs, vis_targets, vis_predictions, args.run_name, epoch)

        # Early stopping
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
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


    # Save the loss curves (training and validation)
    plt.figure()
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_curve_path = os.path.join("results", args.run_name, "loss_curves.jpg")
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
    plt.savefig(loss_curve_path)
    wandb.log({"Loss Curves": wandb.Image(loss_curve_path)})
    wandb.save(loss_curve_path)


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="lucid_blastformer_1")
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=96)
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