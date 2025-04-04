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
from unscaler_cnn import UnscalerCNN

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
    model_1_parameters = sum(p.numel() for p in model_1.parameters() if p.requires_grad)

    model_2 = FNO2d_cond(
        time_window_in=args.time_window, 
        time_window_out=args.time_window_out,
        modes1=args.modes1, 
        modes2=args.modes2, 
        width=args.width, 
        cond_channels=args.cond_channels, 
        num_layers=args.num_layers
    ).to(device)
    model_2_parameters = sum(p.numel() for p in model_2.parameters() if p.requires_grad)

    model_3 = BlastCNN().to(device)
    model_3_parameters = sum(p.numel() for p in model_3.parameters() if p.requires_grad)

    unscaler = UnscalerCNN().to(device)


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
    
    # Load the unscaler model
    checkpoint_path = os.path.join(args.checkpoint_dir, args.unscaling_CNN_run_name)
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading model from {checkpoint_path}")
        unscaler.load_state_dict(torch.load(checkpoint_path))
    else:
        logging.error(f"Checkpoint not found at {checkpoint_path}")
        return

    # Set the model to evaluation mode
    model_1.eval()
    model_2.eval()

    # Initialize lists to store predictions and targets
    all_blastOformer_predictions = []
    all_FNO_predictions = []
    all_CNN_predictions = []
    all_blastOformer_unscaled_predictions = []
    all_FNO_unscaled_predictions = []
    all_CNN_unscaled_predictions = []
    all_targets = []
    all_percent_errors = []
    prediction_times = []
    FNO_prediction_times = []
    CNN_prediction_times = []

    BlastOFormer_mean_percent_errors = []
    FNO_mean_percent_errors = []
    CNN_mean_percent_errors = []

    BlastOformer_MSE_unscaled = []
    FNO_MSE_unscaled = []
    CNN_MSE_unscaled = []
    all_targets_unscaled = []

    BlastOFormer_mae_log = []
    FNO_mae_log = []
    CNN_mae_log = []
    BlastOformer_mae_unscaled = []
    FNO_mae_unscaled = []
    CNN_mae_unscaled = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            max_pressure = batch["max_pressure"].to(device)
            all_targets.append(max_pressure.cpu().numpy())
            max_pressure_unscaled = inverse_transform(max_pressure.detach().cpu(), min_max_pressure, max_max_pressure, normalized=normalize)
            all_targets_unscaled.append(max_pressure_unscaled.cpu().numpy())
            probe_positions = batch["probe_positions"][:, :, :, 0:2].to(device)

            # FNO
            charge_mass = batch["charge_mass"].to(device)
            charge_mass_expanded = charge_mass.view(-1, 1, 1, 1).expand(-1, -1, 99, 99)
            charge_center = batch["charge_center"].to(device)
            wall_1 = batch["wall_1"].to(device)
            wall_2 = batch["wall_2"].to(device)
            wall_3 = batch["wall_3"].to(device)
            conditioning = torch.cat([charge_center, wall_1, wall_2, wall_3], dim=1)
            start_time = time.time()
            FNO_prediction = model_2(charge_mass_expanded, conditioning).squeeze()
            all_FNO_predictions.append(FNO_prediction.unsqueeze(0).cpu().numpy())
            FNO_difference = torch.abs(FNO_prediction - max_pressure)
            FNO_mae_log.append(torch.mean(FNO_difference).cpu().numpy())
            FNO_percent_error = (FNO_difference / max_pressure) * 100
            FNO_mean_percent_error = torch.mean(FNO_percent_error)
            FNO_mean_percent_errors.append(FNO_mean_percent_error.cpu().numpy())
            FNO_prediction_unscaled = inverse_transform(FNO_prediction.detach().cpu(), min_max_pressure, max_max_pressure, normalized=normalize)
            end_time = time.time()
            all_FNO_unscaled_predictions.append(FNO_prediction_unscaled.unsqueeze(0).cpu().numpy())
            FNO_prediction_time = end_time - start_time
            FNO_prediction_times.append(FNO_prediction_time)
            FNO_unscaled_error = torch.abs(FNO_prediction_unscaled - max_pressure_unscaled)
            FNO_mae_unscaled.append(torch.mean(FNO_unscaled_error).cpu().numpy())
            FNO_unscaled_percent_error = (FNO_unscaled_error / max_pressure_unscaled) * 100
            FNO_unscaled_mean_percent_error = torch.mean(FNO_unscaled_percent_error)
            FNO_MSE_unscaled.append(FNO_unscaled_mean_percent_error.cpu().numpy()) 
            

            # BlastOFormer
            x = batch["input_tensor"].to(device)
            y = max_pressure.unsqueeze(-1)
            start_time = time.time()
            prediction = model_1(x, probe_positions)
            all_blastOformer_predictions.append(prediction.squeeze(-1).cpu().numpy())
            prediction_unscaled = inverse_transform(prediction.detach().cpu(), min_max_pressure, max_max_pressure, normalized=normalize)
            prediction_unscaled = unscaler(prediction_unscaled.squeeze(-1).to(device)).detach().cpu()
            end_time = time.time()
            all_blastOformer_unscaled_predictions.append(prediction_unscaled.cpu().numpy())
            prediction_time = end_time - start_time
            prediction_times.append(prediction_time)
            difference = torch.abs(prediction - y)
            BlastOFormer_mae_log.append(torch.mean(difference).cpu().numpy())
            percent_error = (difference / y) * 100
            mean_percent_error = torch.mean(percent_error)
            BlastOFormer_mean_percent_errors.append(mean_percent_error.cpu().numpy())
            unscaled_difference = torch.abs(prediction_unscaled - max_pressure_unscaled)
            BlastOformer_mae_unscaled.append(torch.mean(unscaled_difference).cpu().numpy())
            unscaled_percent_error = (unscaled_difference / max_pressure_unscaled) * 100
            unscaled_mean_percent_error = torch.mean(unscaled_percent_error)
            BlastOformer_MSE_unscaled.append(unscaled_mean_percent_error.cpu().numpy())
            all_percent_errors.append(percent_error.cpu().numpy())

            # CNN
            start_time = time.time()
            CNN_prediction = model_3(x)
            all_CNN_predictions.append(CNN_prediction.squeeze(-1).cpu().numpy())
            CNN_difference = torch.abs(CNN_prediction - max_pressure)
            CNN_mae_log.append(torch.mean(CNN_difference).cpu().numpy())
            CNN_percent_error = (CNN_difference / max_pressure) * 100
            CNN_mean_percent_error = torch.mean(CNN_percent_error)
            CNN_mean_percent_errors.append(CNN_mean_percent_error.cpu().numpy())
            CNN_prediction_unscaled = inverse_transform(CNN_prediction.detach().cpu(), min_max_pressure, max_max_pressure, normalized=normalize)
            end_time = time.time()
            all_CNN_unscaled_predictions.append(CNN_prediction_unscaled.squeeze(-1).cpu().numpy())
            CNN_prediction_time = end_time - start_time
            CNN_prediction_times.append(CNN_prediction_time)
            CNN_unscaled_error = torch.abs(CNN_prediction_unscaled - max_pressure_unscaled)
            CNN_mae_unscaled.append(torch.mean(CNN_unscaled_error).cpu().numpy())
            CNN_unscaled_percent_error = (CNN_unscaled_error / max_pressure_unscaled) * 100
            CNN_unscaled_mean_percent_error = torch.mean(CNN_unscaled_percent_error)
            CNN_MSE_unscaled.append(CNN_unscaled_mean_percent_error.cpu().numpy())

            if visualize:
                # convert to numpy arrays
                # BlastOFormer
                prediction = prediction.squeeze().cpu().numpy()
                max_pressure = max_pressure.squeeze().cpu().numpy()
                print(f"prediction time: {prediction_time:.4f} seconds")
                visualize_testing(max_pressure, prediction, "BlastOFormer", unscaled=False, vmin=0, vmax=2.0)
                # FNO
                FNO_prediction = FNO_prediction.squeeze().cpu().numpy()
                visualize_testing(max_pressure, FNO_prediction, "FNO", unscaled=False, vmin=0, vmax=2.0)

                # CNN
                CNN_prediction = CNN_prediction.squeeze().cpu().numpy()
                visualize_testing(max_pressure, CNN_prediction, "CNN", unscaled=False, vmin=0, vmax=2.0)

                # unscaled
                prediction_unscaled = prediction_unscaled.squeeze().cpu().numpy()
                max_pressure_unscaled = max_pressure_unscaled.squeeze().cpu().numpy()
                visualize_testing(max_pressure_unscaled, prediction_unscaled, "BlastOFormer_unscaled", vmin=0, vmax=3.6e6)

                # FNO
                FNO_prediction_unscaled = FNO_prediction_unscaled.squeeze().cpu().numpy()
                visualize_testing(max_pressure_unscaled, FNO_prediction_unscaled, "FNO_unscaled", vmin=0, vmax=3.6e6)
                # CNN
                CNN_prediction_unscaled = CNN_prediction_unscaled.squeeze().cpu().numpy()
                visualize_testing(max_pressure_unscaled, CNN_prediction_unscaled, "CNN_unscaled", vmin=0, vmax=3.6e6)

    # Create subplots for histograms
    fig, axes = plt.subplots(figsize=(20, 6))

    # Plot histogram for BlastOFormer
    axes.hist(BlastOFormer_mean_percent_errors, bins=10, alpha=0.5, label='BlastOFormer', color='blue')
    axes.hist(FNO_mean_percent_errors, bins=10, alpha=0.5, label='FNO', color='orange')
    axes.hist(CNN_mean_percent_errors, bins=10, alpha=0.5, label='CNN', color='green')
    axes.set_xlabel('Mean Percent Error', fontsize=24)
    axes.set_ylabel('Frequency', fontsize=24)
    axes.set_title('Log Domain Mean Percent Errors', fontsize=26)
    axes.tick_params(axis='both', labelsize=22)
    axes.legend(fontsize=22)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

    # Create subplots for histograms
    fig, axes = plt.subplots(figsize=(20, 6))

    # Plot histogram for BlastOFormer
    axes.hist(BlastOformer_MSE_unscaled, bins=10, alpha=0.5, label='BlastOFormer', color='blue')
    # Plot histogram for FNO and CNN
    axes.hist(FNO_MSE_unscaled, bins=10, alpha=0.5, label='FNO', color='orange')
    axes.hist(CNN_MSE_unscaled, bins=10, alpha=0.5, label='CNN', color='green')
    axes.set_xlabel('Mean Percent Error', fontsize=24)
    axes.set_ylabel('Frequency', fontsize=24)
    axes.set_title('Unscaled Mean Percent Errors', fontsize=26)
    axes.tick_params(axis='both', labelsize=22)
    axes.legend(fontsize=22)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

    # histogram of prediction times
    fig, axes = plt.subplots(figsize=(15, 6))
    axes.hist(prediction_times, bins=10, alpha=0.5, label='BlastOFormer', color='blue')
    axes.hist(FNO_prediction_times, bins=10, alpha=0.5, label='FNO', color='orange')
    axes.hist(CNN_prediction_times, bins=10, alpha=0.5, label='CNN', color='green')
    axes.set_xlabel('Prediction Time (seconds)', fontsize=20)
    axes.set_ylabel('Frequency', fontsize=20)
    axes.set_title('Prediction Times', fontsize=22)
    axes.tick_params(axis='both', labelsize=18)
    axes.legend(fontsize=18)
    plt.show()


    # print R^2
    R2_blastoformer_log = r2_score_total(all_targets, all_blastOformer_predictions)
    R2_FNO_log = r2_score_total(all_targets, all_FNO_predictions)
    R2_CNN_log = r2_score_total(all_targets, all_CNN_predictions)
    print(f'R^2 BlastOFormer: {R2_blastoformer_log:.4f}')
    print(f'R^2 FNO: {R2_FNO_log:.4f}')
    print(f'R^2 CNN: {R2_CNN_log:.4f}')
    # print R^2 unscaled
    R2_blastoformer_unscaled = r2_score_total(all_targets_unscaled, all_blastOformer_unscaled_predictions)
    R2_FNO_unscaled = r2_score_total(all_targets_unscaled, all_FNO_unscaled_predictions)
    R2_CNN_unscaled = r2_score_total(all_targets_unscaled, all_CNN_unscaled_predictions)
    print(f'R^2 BlastOFormer unscaled: {R2_blastoformer_unscaled:.4f}')
    print(f'R^2 FNO unscaled: {R2_FNO_unscaled:.4f}')
    print(f'R^2 CNN unscaled: {R2_CNN_unscaled:.4f}')

    # print mean percent errors
    mean_blastOformer_percent_error = np.mean(BlastOFormer_mean_percent_errors)
    mean_FNO_percent_error = np.mean(FNO_mean_percent_errors)
    mean_CNN_percent_error = np.mean(CNN_mean_percent_errors)
    print(f'Mean percent error BlastOFormer: {mean_blastOformer_percent_error:.4f}%')
    print(f'Mean percent error FNO: {mean_FNO_percent_error:.4f}%')
    print(f'Mean percent error CNN: {mean_CNN_percent_error:.4f}%')
    # print mean unscaled percent errors
    mean_blastOformer_unscaled_percent_error = np.mean(BlastOformer_MSE_unscaled)
    mean_FNO_unscaled_percent_error = np.mean(FNO_MSE_unscaled)
    mean_CNN_unscaled_percent_error = np.mean(CNN_MSE_unscaled)
    print(f'Mean unscaled percent error BlastOFormer: {mean_blastOformer_unscaled_percent_error:.4f}%')
    print(f'Mean unscaled percent error FNO: {mean_FNO_unscaled_percent_error:.4f}%')
    print(f'Mean unscaled percent error CNN: {mean_CNN_unscaled_percent_error:.4f}%')

    # print mean absolute error
    mean_blastOformer_mae_log = np.mean(BlastOFormer_mae_log)
    mean_FNO_mae_log = np.mean(FNO_mae_log)
    mean_CNN_mae_log = np.mean(CNN_mae_log)
    print(f'Mean absolute error BlastOFormer: {mean_blastOformer_mae_log:.4f}')
    print(f'Mean absolute error FNO: {mean_FNO_mae_log:.4f}')
    print(f'Mean absolute error CNN: {mean_CNN_mae_log:.4f}')
    # print mean unscaled absolute error
    mean_blastOformer_unscaled_mae = np.mean(BlastOformer_mae_unscaled)
    mean_FNO_unscaled_mae = np.mean(FNO_mae_unscaled)
    mean_CNN_unscaled_mae = np.mean(CNN_mae_unscaled)
    print(f'Mean unscaled absolute error BlastOFormer: {mean_blastOformer_unscaled_mae:.4f}')
    print(f'Mean unscaled absolute error FNO: {mean_FNO_unscaled_mae:.4f}')
    print(f'Mean unscaled absolute error CNN: {mean_CNN_unscaled_mae:.4f}')
    # print mean prediction times
    mean_prediction_time = np.mean(prediction_times)
    mean_FNO_prediction_time = np.mean(FNO_prediction_times)
    mean_CNN_prediction_time = np.mean(CNN_prediction_times)
    print(f'Mean prediction time BlastOFormer: {mean_prediction_time:.4f} seconds')
    print(f'Mean prediction time FNO: {mean_FNO_prediction_time:.4f} seconds')
    print(f'Mean prediction time CNN: {mean_CNN_prediction_time:.4f} seconds')

    # print number of parameters
    print(f'Number of parameters BlastOFormer: {model_1_parameters}')
    print(f'Number of parameters FNO: {model_2_parameters}')
    print(f'Number of parameters CNN: {model_3_parameters}')



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

    # Unscaling CNN parameters
    parser.add_argument("--unscaling_CNN_run_name", type=str, default="Unscaler_CNN_home.pt")

    args = parser.parse_args()

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    launch()
