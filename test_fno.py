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
from fno import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def test(args):
    setup_logging(args.run_name)
    device = args.device

    # Load test dataset
    dataset = BlastDataset(args.dataset_path, split="test", normalize=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(16, os.cpu_count() - 1))
    

    # Load model
    model = FNO2d_cond(
        time_window=args.time_window,
        modes1=args.modes1,
        modes2=args.modes2,
        width=args.width,
        cond_channels=args.cond_channels,
        num_layers=args.num_layers
    ).to(device)

    model_path = os.path.join("models", args.run_name, "best_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logging.info(f"Loaded model from {model_path}")
    else:
        logging.error(f"Model checkpoint not found at {model_path}")
        return
    
    model.eval()  # Set model to evaluation mode


    simulation_to_visualize = 1060

    sample_embedding = []
    sample_times = []
    sample_pressures = []

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
                wall_locations = wall_locations.reshape(wall_locations.shape[0], -1)
                current_time = batch["source_time"].to(device)
                cond_emb = torch.cat([charge_data, current_time, wall_locations], dim=1)
                sample_embedding.extend([e for e in cond_emb])
                sample_pressures.extend([p for p in current_pressure])
                sample_times.extend([t for t in current_time])

            if len(sample_embedding) >= 500:
                break
    clipped_sample_pressures = []
    last_pressure = None
    with torch.no_grad():
        for i,time in enumerate(sample_times):
            print(f"Processing sample {i}, time {time}")
            if i >= 0:
                clipped_sample_pressures.append(sample_pressures[i])
                if last_pressure is None:
                    last_pressure = sample_pressures[i]
                    print(f"last_pressure shape: {last_pressure.shape}")
                    print(f"sample_embedding shape: {sample_embedding[i].shape}")
                    predicted_pressure = model(last_pressure.unsqueeze(0).unsqueeze(0), sample_embedding[i].unsqueeze(0)).squeeze(1)
                    predicted_pressures.append(predicted_pressure.squeeze(0))
                    last_pressure = predicted_pressure
                else:
                    predicted_pressure = model(last_pressure.unsqueeze(0), sample_embedding[i].unsqueeze(0)).squeeze(1)
                    predicted_pressures.append(predicted_pressure.squeeze(0))
                    last_pressure = predicted_pressure

    

    plot_recursive_predictions(clipped_sample_pressures, predicted_pressures, sample_times)





def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="fno_home_24_width_run")
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    # New model-specific arguments:
    parser.add_argument('--time_window', type=int, default=1, help="Number of channels for pressure input")
    parser.add_argument('--modes1', type=int, default=6)
    parser.add_argument('--modes2', type=int, default=6)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--cond_channels', type=int, default=26, help="Dimension of conditioning embedding (matches conditioning dimension[1])")
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dataset_path', type=str, default="/home/reid/projects/blast_waves/hdf5_dataset")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    wandb.init(project="blastformer_test", name=f"{args.run_name}_test")
    test(args)


if __name__ == '__main__':
    launch()

