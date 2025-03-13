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
# Import the new model. Adjust the import path as needed.
from fno import FNO2d_cond

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(args):
    setup_logging(args.run_name)
    device = args.device

    training_dataset = BlastDataset(args.dataset_path, split="train", normalize=True)
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(16, os.cpu_count() - 1))
    l = len(training_dataloader)

    validation_dataset = BlastDataset(args.dataset_path, split="val", normalize=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(16, os.cpu_count() - 1))
    if len(validation_dataloader) == 0:
        logging.error("Validation dataloader is empty. Check the dataset path.")
        return

    # Instantiate the new FNO2d_cond model.
    # For this model, we assume that:
    # - the pressure field is 2D (shape: [batch, 99, 99]) and we unsqueeze to get a channel dimension,
    # - we use charge_data as the conditioning signal (averaged over its time dimension to yield shape [batch, cond_channels]),
    # - time_window is set to 1 (since the input pressure is a single channel).
    model = FNO2d_cond(
        time_window_in=args.time_window, 
        time_window_out=args.time_window,
        modes1=args.modes1, 
        modes2=args.modes2, 
        width=args.width, 
        cond_channels=args.cond_channels, 
        num_layers=args.num_layers
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    # Wandb setup
    wandb.init(project='blastformer_max_pressure_FNO', name=args.run_name, config=args)
    config = wandb.config
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr

    training_loss = []
    validation_loss = []

    best_loss = float('inf')
    patience = args.patience
    epochs_no_improve = 0

    steps_forward = 1

    grid_size = args.grid_size

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(training_dataloader)
        epoch_train_loss = 0
        if epoch > args.warmup_epochs and steps_forward < 9: # make sure we don't go over 10 steps forward
            steps_forward += 1

        model.train()
        for i, batch in enumerate(pbar):
            charge_mass = batch["charge_mass"].to(device)
            charge_mass_expanded = charge_mass.view(-1, 1, 1, 1).expand(-1, -1, grid_size, grid_size)
            charge_center = batch["charge_center"].to(device)
            wall_1 = batch["wall_1"].to(device)
            wall_2 = batch["wall_2"].to(device)
            wall_3 = batch["wall_3"].to(device)
            max_pressure = batch["max_pressure"].to(device)
            conditioning = torch.cat([charge_center, wall_1, wall_2, wall_3], dim=1)
            # make grid with charge mass in one channel and charge center in the other
            model_input = charge_mass_expanded

            outputs = model(model_input,conditioning)
            loss = l1(outputs.squeeze(), max_pressure)
            scaled_loss = scaledlp_loss(outputs.squeeze(), max_pressure, p=2, reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

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

        epoch_train_loss /= len(training_dataloader)
        training_loss.append(epoch_train_loss)
        scheduler.step()

        # Validation
        model.eval()
        eval_model_loss = 0
        # store the first batch for visualization
        vis_inputs, vis_targets, vis_predictions = None, None, None
        with torch.no_grad():
            for j, val_batch in enumerate(validation_dataloader):
                val_charge_center = val_batch["charge_center"].to(device)
                val_charge_mass = val_batch["charge_mass"].to(device)
                val_charge_mass_expanded = val_charge_mass.view(-1, 1, 1, 1).expand(-1, -1, grid_size, grid_size)
                val_wall_1 = val_batch["wall_1"].to(device)
                val_wall_2 = val_batch["wall_2"].to(device)
                val_wall_3 = val_batch["wall_3"].to(device)
                val_pressures = val_batch["pressures"].to(device)
                val_max_pressure = val_batch["max_pressure"].to(device)
                val_current_pressure = val_pressures[:, 0, :, :].unsqueeze(1)
                val_conditioning = torch.cat([val_charge_center, val_wall_1, val_wall_2, val_wall_3, ], dim=1)
                model_input = val_charge_mass_expanded
                val_predicted_pressure = model(model_input, val_conditioning)
                val_loss = l1(val_predicted_pressure.squeeze(), val_max_pressure)
                eval_model_loss += val_loss.item()

                if j == 0:
                    vis_inputs = val_current_pressure
                    vis_targets = val_max_pressure
                    vis_predictions = val_predicted_pressure

            epoch_val_loss = eval_model_loss / len(validation_dataloader)
            validation_loss.append(epoch_val_loss)
            wandb.log({
                "Validation Loss": epoch_val_loss,
                "Epoch": epoch
            })
            logger.add_scalar("validation_loss", epoch_val_loss, global_step=epoch)
            logging.info(f"Epoch {epoch} - Training Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}")

        # visualize validation predictions
        if vis_inputs is not None:
            print(f'vis_inputs shape: {vis_inputs.shape}')
            print(f'vis_targets shape: {vis_targets.shape}')
            print(f'vis_predictions shape: {vis_predictions.shape}')
            visualize_results(vis_inputs[0][0], vis_targets[0], vis_predictions[0][0], args.run_name, epoch)

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
    parser.add_argument('--run_name', type=str, default="fno_lab_max_pressure")
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--time_window', type=int, default=1, help="Number of channels for pressure input")
    parser.add_argument('--modes1', type=int, default=6)
    parser.add_argument('--modes2', type=int, default=6)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--cond_channels', type=int, default=21, help="Dimension of conditioning embedding (matches conditioning dimension[1])")
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dataset_path', type=str, default="/home/reid/projects/blast_waves/hdf5_dataset_low_res_1_simulation_per_file_10_chunks")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grid_size', type=int, default=99)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    launch()
