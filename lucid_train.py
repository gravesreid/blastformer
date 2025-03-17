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
from hdf5_dataset_new import *
from blastformer_transformer import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def train(args):
    setup_logging(args.run_name)
    device = args.device

    training_dataset = BlastDataset(args.dataset_path, split="train", normalize=True)
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(16, os.cpu_count() - 1))
    l = len(training_dataloader) # used for logging

    validation_dataset = BlastDataset(args.dataset_path, split="val", normalize=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(16, os.cpu_count() - 1))
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
        num_predicted = 0

        for i, batch in enumerate(pbar):
            # Move inputs and targets to device
            charge_center = batch["charge_center"].to(device)
            charge_mass = batch["charge_mass"].to(device)
            wall_1 = batch["wall_1"].to(device)
            wall_2 = batch["wall_2"].to(device)
            wall_3 = batch["wall_3"].to(device)
            times = batch["times"].to(device)
            pressures = batch["pressures"].to(device) # shape: (batch_size, 99,99)

            num_batch_predictions = pressures.shape[1] - 1
            num_predicted += num_batch_predictions
            batch_loss_sum = 0.0
            for t in range(pressures.shape[1] - 1):
                current_pressure = pressures[:, t, :, :].unsqueeze(1)
                next_pressures = pressures[:, t + 1, :, :].unsqueeze(1)
                current_time = times[:, t].unsqueeze(1)
                charge_data = torch.cat([charge_center, charge_mass.unsqueeze(1)], dim=1)
                wall_locations = torch.cat([wall_1, wall_2, wall_3], dim=1)
                predicted_pressure = model(current_pressure, charge_data, wall_locations, current_time)
                loss = l1(predicted_pressure, next_pressures)
                scaled_loss = scaledlp_loss(predicted_pressure, next_pressures, p=2, reduction="mean")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                batch_loss_sum += loss.item()

                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(epoch_loss=loss.item(), scaled_loss=scaled_loss.item(), learning_rate=current_lr)
                wandb.log({
                    "Batch Loss": loss.item(),
                    "scaled_loss": scaled_loss.item(),
                    "Learning Rate": current_lr,
                    "Epoch": epoch
                })
                logger.add_scalar(f"loss: {epoch}", loss.item(), global_step=epoch * l + i)
                logger.add_scalar("learning_rate", current_lr, global_step=epoch * l + i)
            
            avg_batch_loss = batch_loss_sum / num_batch_predictions
            wandb.log({
                "Average Batch Loss": avg_batch_loss,
                "Epoch": epoch
            })
            logger.add_scalar("average_batch_loss", avg_batch_loss, global_step=epoch * l + i)

        epoch_train_loss /= num_predicted
        training_loss.append(epoch_train_loss)
        scheduler.step(epoch_train_loss)

        # Validation
        model.eval()
        eval_model_loss = 0
        eval_scaled_loss = 0
        num_predicted_val = 0

        # Store the first batch for visualization
        vis_inputs, vis_targets, vis_predictions = None, None, None

        with torch.no_grad():
            for j, val_batch in enumerate(validation_dataloader):
                val_charge_center = val_batch["charge_center"].to(device)
                val_charge_mass = val_batch["charge_mass"].to(device)
                val_wall_1 = val_batch["wall_1"].to(device)
                val_wall_2 = val_batch["wall_2"].to(device)
                val_wall_3 = val_batch["wall_3"].to(device)
                val_times = val_batch["times"].to(device)
                val_pressures = val_batch["pressures"].to(device)  # shape: (batch_size, 99, 99)

                num_batch_predictions = val_pressures.shape[1] - 1
                num_predicted_val += num_batch_predictions
                batch_loss_sum = 0.0
                batch_scaled_loss_sum = 0.0

                for t in range(val_pressures.shape[1] - 1):
                    val_current_pressure = val_pressures[:, t, :, :].unsqueeze(1)
                    val_next_pressures = val_pressures[:, t + 1, :, :].unsqueeze(1)
                    val_current_time = val_times[:, t].unsqueeze(1)
                    val_charge_data = torch.cat([val_charge_center, val_charge_mass.unsqueeze(1)], dim=1)
                    val_wall_locations = torch.cat([val_wall_1, val_wall_2, val_wall_3], dim=1)

                    val_predicted_pressure = model(val_current_pressure, val_charge_data, val_wall_locations, val_current_time)

                    val_loss = l1(val_predicted_pressure, val_next_pressures)
                    scaled_val_loss = scaledlp_loss(val_predicted_pressure, val_next_pressures, p=2, reduction="mean")

                    eval_model_loss += val_loss.item()
                    eval_scaled_loss += scaled_val_loss.item()
                    batch_loss_sum += val_loss.item()
                    batch_scaled_loss_sum += scaled_val_loss.item()

                    # Store first batch for visualization
                    if j == 0 and t == 0:
                        vis_inputs = val_current_pressure
                        vis_targets = val_next_pressures
                        vis_predictions = val_predicted_pressure
                        print(f"Visualizing first batch of validation predictions.")
                        print(f"Inputs shape: {vis_inputs.shape}, Targets shape: {vis_targets.shape}, Predictions shape: {vis_predictions.shape}")

                avg_batch_loss = batch_loss_sum / num_batch_predictions
                avg_batch_scaled_loss = batch_scaled_loss_sum / num_batch_predictions

                wandb.log({
                    "Validation Batch Loss": avg_batch_loss,
                    "Validation Scaled Loss": avg_batch_scaled_loss,
                    "Epoch": epoch
                })
                logger.add_scalar("validation_batch_loss", avg_batch_loss, global_step=epoch)

        epoch_val_loss = eval_model_loss / num_predicted_val
        epoch_val_scaled_loss = eval_scaled_loss / num_predicted_val
        validation_loss.append(epoch_val_loss)

        wandb.log({
            "Validation Loss": epoch_val_loss,
            "Validation Scaled Loss": epoch_val_scaled_loss,
            "Epoch": epoch
        })
        logger.add_scalar("validation_loss", epoch_val_loss, global_step=epoch)
        logger.add_scalar("validation_scaled_loss", epoch_val_scaled_loss, global_step=epoch)

        logging.info(f"Epoch {epoch} - Training Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}, Validation Scaled Loss: {epoch_val_scaled_loss}")

        # Visualize validation predictions
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
            return

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
    parser.add_argument('--run_name', type=str, default="lucid_blastformer_lab_ultra_low_res")
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--patch_size', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=302)
    parser.add_argument('--dataset_path', type=str, default="/home/reid/projects/blast_waves/hdf5_dataset_ultra_low_res_1_simulation_per_file")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    launch()