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
from BlastOFormer import BlastOFormer
from fno import FNO2d_cond
from CNN import *
import time

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
\
def test(args):
    device = args.device
    visualize = False
    normalize = False

    test_dataset = BlastDataset(args.dataset_path, split="test", standardize=False, normalize=normalize)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(16, os.cpu_count() - 1))

    min_max_pressure = test_dataset.min_max_pressure
    max_max_pressure = test_dataset.max_max_pressure


    model_1 = BlastOFormer(
        encoder_input_channels=4,
        encoder_in_emb_dim=96,
        encoder_out_seq_emb_dim=256,
        encoder_heads=4,
        encoder_depth=6,
        encoder_res=99,
        decoder_latent_channels=256,
        decoder_out_channels=1,
        decoder_res=99,
        decoder_scale=0.5,
        input_channels=4,
        patch_size=args.patch_size,
        img_size=99
    ).to(device)

    model_2 = FNO2d_cond(
        time_window_in=args.time_window, 
        time_window_out=args.time_window_out,
        modes1=args.modes1, 
        modes2=args.modes2, 
        width=args.width, 
        cond_channels=args.cond_channels, 
        num_layers=args.num_layers
    ).to(device)

    model_3 = BlastCNN().to(device)


    # Load the model checkpoint
    # Load the BlastOFormer model
    checkpoint_path = os.path.join(args.checkpoint_dir, args.run_name)
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading model from {checkpoint_path}")
        model_1.load_state_dict(torch.load(checkpoint_path))
    else:
        logging.error(f"Checkpoint not found at {checkpoint_path}")
        return
    # Load the FNO model
    checkpoint_path = os.path.join(args.checkpoint_dir, args.FNO_run_name)
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading model from {checkpoint_path}")
        model_2.load_state_dict(torch.load(checkpoint_path))
    else:
        logging.error(f"Checkpoint not found at {checkpoint_path}")
        return
    
    # Load the CNN model
    checkpoint_path = os.path.join(args.checkpoint_dir, args.CNN_run_name)
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading model from {checkpoint_path}")
        model_3.load_state_dict(torch.load(checkpoint_path))
    else:
        logging.error(f"Checkpoint not found at {checkpoint_path}")
        return

    # Set the model to evaluation mode
    model_1.eval()
    model_2.eval()

    # Initialize lists to store predictions and targets
    all_predictions = []
    all_targets = []
    all_percent_errors = []
    prediction_times = []

    BlastOFormer_mean_percent_errors = []
    FNO_mean_percent_errors = []
    CNN_mean_percent_errors = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            max_pressure = batch["max_pressure"].to(device)
            probe_positions = batch["probe_positions"][:, :, :, 0:2].to(device)

            # FNO
            charge_mass = batch["charge_mass"].to(device)
            charge_mass_expanded = charge_mass.view(-1, 1, 1, 1).expand(-1, -1, 99, 99)
            charge_center = batch["charge_center"].to(device)
            wall_1 = batch["wall_1"].to(device)
            wall_2 = batch["wall_2"].to(device)
            wall_3 = batch["wall_3"].to(device)
            conditioning = torch.cat([charge_center, wall_1, wall_2, wall_3], dim=1)

            FNO_prediction = model_2(charge_mass_expanded, conditioning).squeeze()
            FNO_difference = torch.abs(FNO_prediction - max_pressure)
            FNO_percent_error = (FNO_difference / max_pressure) * 100
            FNO_mean_percent_error = torch.mean(FNO_percent_error)
            FNO_mean_percent_errors.append(FNO_mean_percent_error.cpu().numpy())
            print(f'FNO Mean percent error: {FNO_mean_percent_error:.4f}%')
            FNO_prediction_unscaled = inverse_transform(FNO_prediction.detach().cpu(), min_max_pressure, max_max_pressure, normalized=normalize)
            

            # BlastOFormer
            x = batch["input_tensor"].to(device)
            y = max_pressure.unsqueeze(-1)
            start_time = time.time()
            prediction = model_1(x, probe_positions)
            end_time = time.time()
            prediction_unscaled = inverse_transform(prediction.detach().cpu(), min_max_pressure, max_max_pressure, normalized=normalize)
            max_pressure_unscaled = inverse_transform(max_pressure.detach().cpu(), min_max_pressure, max_max_pressure, normalized=normalize)
            prediction_time = end_time - start_time
            prediction_times.append(prediction_time)
            difference = torch.abs(prediction - y)
            percent_error = (difference / y) * 100
            mean_percent_error = torch.mean(percent_error)
            BlastOFormer_mean_percent_errors.append(mean_percent_error.cpu().numpy())
            print(f'Mean percent error: {mean_percent_error:.4f}%')
            unscaled_difference = torch.abs(prediction_unscaled - max_pressure_unscaled)
            unscaled_percent_error = (unscaled_difference / max_pressure_unscaled) * 100
            unscaled_mean_percent_error = torch.mean(unscaled_percent_error)
            print(f'Unscaled mean percent error: {unscaled_mean_percent_error:.4f}%')
            all_predictions.append(prediction.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_percent_errors.append(percent_error.cpu().numpy())

            # CNN
            CNN_prediction = model_3(x)
            CNN_difference = torch.abs(CNN_prediction - max_pressure)
            CNN_percent_error = (CNN_difference / max_pressure) * 100
            CNN_mean_percent_error = torch.mean(CNN_percent_error)
            CNN_mean_percent_errors.append(CNN_mean_percent_error.cpu().numpy())
            print(f'CNN Mean percent error: {CNN_mean_percent_error:.4f}%')
            CNN_prediction_unscaled = inverse_transform(CNN_prediction.detach().cpu(), min_max_pressure, max_max_pressure, normalized=normalize)

            if visualize:
                # convert to numpy arrays
                # BlastOFormer
                prediction = prediction.squeeze().cpu().numpy()
                max_pressure = max_pressure.squeeze().cpu().numpy()
                print(f"prediction time: {prediction_time:.4f} seconds")
                visualize_testing(max_pressure, prediction, "BlastOFormer")
                # FNO
                FNO_prediction = FNO_prediction.squeeze().cpu().numpy()
                visualize_testing(max_pressure, FNO_prediction, "FNO")

                # CNN
                CNN_prediction = CNN_prediction.squeeze().cpu().numpy()
                visualize_testing(max_pressure, CNN_prediction, "CNN")

                # unscaled
                prediction_unscaled = prediction_unscaled.squeeze().cpu().numpy()
                max_pressure_unscaled = max_pressure_unscaled.squeeze().cpu().numpy()
                visualize_testing(max_pressure_unscaled, prediction_unscaled, "BlastOFormer_unscaled")

                # FNO
                FNO_prediction_unscaled = FNO_prediction_unscaled.squeeze().cpu().numpy()
                visualize_testing(max_pressure_unscaled, FNO_prediction_unscaled, "FNO_unscaled")
                # CNN
                CNN_prediction_unscaled = CNN_prediction_unscaled.squeeze().cpu().numpy()
                visualize_testing(max_pressure_unscaled, CNN_prediction_unscaled, "CNN_unscaled")

    # plot histogram of percent errors
    plt.figure(figsize=(10, 6))
    plt.hist(BlastOFormer_mean_percent_errors, bins=10, alpha=0.5, label='BlastOFormer')
    plt.hist(FNO_mean_percent_errors, bins=10, alpha=0.5, label='FNO')
    plt.hist(CNN_mean_percent_errors, bins=10, alpha=0.5, label='CNN')
    plt.xlabel('Mean Percent Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Mean Percent Errors')
    plt.legend()
    plt.show()


def launch():
    parser = argparse.ArgumentParser()
    # BlastOFormer parameters
    parser.add_argument("--dataset_path", type=str, default="/home/reid/projects/blast_waves/hdf5_dataset_max_pressure")
    parser.add_argument("--run_name", type=str, default="BlastOFormer_Home_og_dataset_l1_loss_log_only.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", type=str, default="/home/reid/projects/blast_waves/blastformer/models")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=3)

    # FNO parameters
    parser.add_argument("--FNO_run_name", type=str, default="FNO_Home_OG_dataset_l1_loss_log.pt")
    parser.add_argument('--time_window', type=int, default=1, help="Number of channels for pressure input")
    parser.add_argument('--time_window_out', type=int, default=1, help="Number of channels for pressure output")
    parser.add_argument('--modes1', type=int, default=6)
    parser.add_argument('--modes2', type=int, default=6)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--cond_channels', type=int, default=21, help="Dimension of conditioning embedding (matches conditioning dimension[1])")
    parser.add_argument('--num_layers', type=int, default=4)

    # CNN parameters
    parser.add_argument("--CNN_run_name", type=str, default="CNN_Home_OG_dataset_l1_loss_log.pt")

    args = parser.parse_args()

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    launch()
