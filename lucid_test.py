import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from utils import *
from hdf5_dataset import *
from blastformer_transformer import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def test(args):
    setup_logging(args.run_name)
    device = args.device

    # Load test dataset
    dataset = BlastDataset(args.dataset_path, split="test", normalize=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(16, os.cpu_count() - 1))
    
    patch_size = args.patch_size
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    output_dim = 99
    input_dim = (99**2) // (patch_size**2)

    # Load model
    model = BlastFormer(input_dim, hidden_dim, num_layers, output_dim, patch_size).to(device)
    model_path = os.path.join("models", args.run_name, "best_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logging.info(f"Loaded model from {model_path}")
    else:
        logging.error(f"Model checkpoint not found at {model_path}")
        return
    
    model.eval()  # Set model to evaluation mode

    l1 = nn.L1Loss()
    l2 = nn.MSELoss()

    total_l1_loss = 0
    total_l2_loss = 0

    simulation_to_visualize = 1057

    sample_pressures = []
    sample_times = []
    sample_charge_data = []
    sample_wall_locations = []

    predicted_pressures = []

    # Run inference
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing")
        for batch in pbar:
            # Move inputs to device
            simulation_numbers = batch["simulation_number"]
            if simulation_to_visualize not in simulation_numbers:
                continue
            else:
                current_pressure = batch["source_pressure"].to(device)  # shape: (batch_size, 99, 99)
                charge_data = batch["source_charge_data"].to(device)
                wall_locations = batch["source_wall_locations"].to(device)
                current_time = batch["source_time"].to(device)
                sample_pressures.extend([p for p in current_pressure])
                sample_times.extend([t for t in current_time])
                sample_charge_data.extend([c for c in charge_data])
                sample_wall_locations.extend([w for w in wall_locations])

            if len(sample_times) >= 800:
                break
    clipped_sample_pressures = []
    clipped_times = []
    last_pressure = None
    l1_losses = []
    l2_losses = []
    with torch.no_grad():
        for i,time in enumerate(sample_times):
            print(f"Processing sample {i}, time {time}")
            if i > 100:
                clipped_sample_pressures.append(sample_pressures[i])
                clipped_times.append(sample_times[i])
                if last_pressure is None:
                    last_pressure = sample_pressures[i]
                    predicted_pressure = model(last_pressure.unsqueeze(0), sample_charge_data[i].unsqueeze(0), sample_wall_locations[i].unsqueeze(0), time.unsqueeze(0)).squeeze(1)
                    predicted_pressures.append(predicted_pressure.squeeze(0))
                    l1_loss = l1(predicted_pressure, sample_pressures[i])
                    l2_loss = l2(predicted_pressure, sample_pressures[i])
                    l1_losses.append(l1_loss.item())
                    l2_losses.append(l2_loss.item())
                    last_pressure = predicted_pressure
                else:
                    last_pressure = sample_pressures[i]
                    predicted_pressure = model(last_pressure.unsqueeze(0), sample_charge_data[i].unsqueeze(0), sample_wall_locations[i].unsqueeze(0), time.unsqueeze(0)).squeeze(1)
                    predicted_pressures.append(predicted_pressure.squeeze(0))
                    l1_loss = l1(predicted_pressure, sample_pressures[i])
                    l2_loss = l2(predicted_pressure, sample_pressures[i])
                    l1_losses.append(l1_loss.item())
                    l2_losses.append(l2_loss.item())
                    last_pressure = predicted_pressure

    

    plot_recursive_predictions(clipped_sample_pressures, predicted_pressures, clipped_times)


    # plot losses
    plt.figure()
    plt.plot(l1_losses, label="L1 Loss")
    plt.plot(l2_losses, label="L2 Loss")
    plt.legend()
    plt.show()





def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="lucid_blastformer_lab-512_hidden_dim")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dataset_path', type=str, default="/home/reid/projects/blast_waves/hdf5_dataset")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()
    
    wandb.init(project="blastformer_test", name=f"{args.run_name}_test")
    test(args)


if __name__ == '__main__':
    launch()
