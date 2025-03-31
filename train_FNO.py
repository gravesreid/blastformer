import os
import numpy as np
import argparse
from einops import rearrange
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import logging
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import wandb
import matplotlib.pyplot as plt
from utils import *
from hdf5_dataset_max_pressure import *
from fno import FNO2d_cond


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def train(args):
    setup_logging(args.run_name)
    device = args.device
    normalize = True

    training_dataset = BlastDataset(args.dataset_path, split="train", standardize=False, normalize=normalize)
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(16, os.cpu_count() - 1))
    l = len(training_dataloader)

    min_max_pressure = training_dataset.min_max_pressure
    max_max_pressure = training_dataset.max_max_pressure

    validation_dataset = BlastDataset(args.dataset_path, split="val", standardize=False, normalize=normalize)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(16, os.cpu_count() - 1))
    if len(validation_dataloader) == 0:
        logging.error("Validation dataloader is empty. Check the dataset path.")
        return
    
    model = FNO2d_cond(
        time_window_in=args.time_window, 
        time_window_out=args.time_window_out,
        modes1=args.modes1, 
        modes2=args.modes2, 
        width=args.width, 
        cond_channels=args.cond_channels, 
        num_layers=args.num_layers
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    #scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1000, verbose=True)
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    wandb.init(project="FNO Blast Pressure Predictor", name=args.run_name, config=args)
    config = wandb.config
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    grid_size = args.grid_size

    training_loss = []
    validation_loss = []

    best_loss = float('inf')
    patience = args.patience
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        pbar = tqdm(training_dataloader, desc="Training", total=l)
        model.train()
        epoch_training_loss = 0
        epoch_scaled_loss = 0
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            max_pressure = batch["max_pressure"].to(device)
            charge_mass = batch["charge_mass"].to(device)
            charge_mass_expanded = charge_mass.view(-1, 1, 1, 1).expand(-1, -1, grid_size, grid_size)
            charge_center = batch["charge_center"].to(device)
            wall_1 = batch["wall_1"].to(device)
            wall_2 = batch["wall_2"].to(device)
            wall_3 = batch["wall_3"].to(device)
            conditioning = torch.cat([charge_center, wall_1, wall_2, wall_3], dim=1)
            #x = rearrange(x,"b w h c -> b (w h) c")
            y = max_pressure
            prediction = model(charge_mass_expanded, conditioning).squeeze()
            loss = l1(prediction, y)
            l2_loss = l2(prediction, y)
            scaled_loss = scaledlp_loss(prediction, y, p=2, reduction="mean")
            loss.backward()
            #scaled_loss.backward()
            #l2_loss.backward()
            optimizer.step()
            pbar.set_postfix({"Loss": loss.item(), "scaled_loss": scaled_loss.item()})
            wandb.log({"train/loss": loss.item()})
            logger.add_scalar("Loss/train", loss.item(), epoch * l + i)
            epoch_training_loss += loss.item()
            epoch_scaled_loss += scaled_loss.item()

        scheduler.step()
        epoch_training_loss /= len(training_dataloader)
        epoch_scaled_loss /= len(training_dataloader)
        training_loss.append(epoch_training_loss)
        current_LR = optimizer.param_groups[0]['lr']
        logging.info(f"Learning rate: {current_LR}")
        logging.info(f"Training loss: {epoch_training_loss}")
        wandb.log({"train/epoch_loss": epoch_training_loss, "train/scaled_epoch_loss": epoch_scaled_loss, "lr": current_LR})

        val_pbar = tqdm(validation_dataloader, desc="Validation")
        model.eval()
        epoch_val_loss = 0
        epoch_val_scaled_loss = 0
        with torch.no_grad():
            for j, batch in enumerate(val_pbar):
                charge_mass = batch["charge_mass"].to(device)
                charge_mass_expanded = charge_mass.view(-1, 1, 1, 1).expand(-1, -1, grid_size, grid_size)
                charge_center = batch["charge_center"].to(device)
                wall_1 = batch["wall_1"].to(device)
                wall_2 = batch["wall_2"].to(device)
                wall_3 = batch["wall_3"].to(device)
                max_pressure = batch["max_pressure"].to(device)
                probe_positions = batch["probe_positions"][:,:,:,0:2].to(device)
                x = batch["input_tensor"].to(device)
                conditioning = torch.cat([charge_center, wall_1, wall_2, wall_3], dim=1)
                y = max_pressure
                prediction = model(charge_mass_expanded, conditioning).squeeze()
                loss = l1(prediction, y)
                scaled_loss = scaledlp_loss(prediction, y, p=2, reduction="mean")
                epoch_val_loss += loss.item()
                val_pbar.set_postfix({"Loss": loss.item()})
                logger.add_scalar("Loss/val", loss.item(), epoch * len(validation_dataloader) + j)

                epoch_val_scaled_loss += scaled_loss.item()

                if j == 0:
                    target = y
                    model_prediction = prediction

        epoch_val_loss /= len(validation_dataloader)
        epoch_val_scaled_loss /= len(validation_dataloader)
        validation_loss.append(epoch_val_loss)
        logging.info(f"Validation loss: {epoch_val_loss}")
        wandb.log({"val/epoch_loss": epoch_val_loss, "val/scaled_epoch_loss": epoch_val_scaled_loss})

        if target is not None:
            print(f"Visualizing max pressure predictions for epoch {epoch}, run {args.run_name}")
            visualize_max_pressure(target, model_prediction, args.run_name, epoch)
            # also visualize for original scaled values
            target_unscaled = inverse_transform(target.detach().cpu(), min_max_pressure, max_max_pressure, normalized=normalize)
            model_prediction_unscaled = inverse_transform(model_prediction.detach().cpu(), min_max_pressure, max_max_pressure, normalized=normalize)
            print(f"Visualizing max pressure predictions for epoch {epoch}, run {args.run_name} (unscaled)")
            visualize_max_pressure(target_unscaled, model_prediction_unscaled, args.run_name + "_unscaled", epoch)

        if epoch_val_loss < best_loss:
            print(f'Validation loss decreased from {best_loss:.4f} to {epoch_val_loss:.4f}. Saving model...')
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), f"models/{args.run_name}.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping after {epoch} epochs.')
                break



def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/reid/projects/blast_waves/hdf5_dataset_max_pressure")
    parser.add_argument("--run_name", type=str, default="FNO_Home_OG_dataset_l1_loss_log_normalized")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--patch_size", type=int, default=3)
    parser.add_argument('--time_window', type=int, default=1, help="Number of channels for pressure input")
    parser.add_argument('--time_window_out', type=int, default=1, help="Number of channels for pressure output")
    parser.add_argument('--modes1', type=int, default=6)
    parser.add_argument('--modes2', type=int, default=6)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--cond_channels', type=int, default=21, help="Dimension of conditioning embedding (matches conditioning dimension[1])")
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--grid_size', type=int, default=99)
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    launch()
