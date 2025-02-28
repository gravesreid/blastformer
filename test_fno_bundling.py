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
from hdf5_dataset_new import *
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
        time_window_in=args.time_window,
        time_window_out=args.time_window_out,
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


    simulation_to_visualize = 1091

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
            print(f"simulation_numbers: {simulation_numbers}")
            if simulation_to_visualize not in simulation_numbers:
                continue
            else:
                charge_center = batch["charge_center"].to(device)
                charge_mass = batch["charge_mass"].to(device)
                wall_1 = batch["wall_1"].to(device)
                wall_2 = batch["wall_2"].to(device)
                wall_3 = batch["wall_3"].to(device)
                times = batch["times"].to(device)
                cond_emb = torch.cat((charge_center, charge_mass.unsqueeze(1), wall_1, wall_2, wall_3, times[:, :-1]), dim=1)
                pressures = batch["pressures"].to(device)
                print(f"pressures: {pressures.shape}")
                sample_embedding.extend([e for e in cond_emb])
                sample_pressures.extend([p for p in pressures[:, -1, :, :]])
                sample_times.extend([t for t in times[:, -1]])

            if len(sample_embedding) >= 890:
                break
    clipped_sample_pressures = []
    times_list = []
    last_pressure = None
    index_to_start = 60
    print(f'sample_pressures: {len(sample_pressures)}')
    print(f'sample pressures sample: {sample_pressures[0].shape}')
    initial_pressure = torch.stack(sample_pressures[index_to_start:index_to_start+9])
    current_pressure = initial_pressure.unsqueeze(0)
    predicted_pressures = []
    fig, ax = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"Simulation {simulation_to_visualize} - Recursive Predictions")
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(index_to_start, len(sample_embedding), 1):
            clipped_sample_pressures.append(sample_pressures[i])
            conditioning = sample_embedding[i].unsqueeze(0)
            current_time = sample_times[i].cpu().numpy()
            predicted_pressure = model(current_pressure, conditioning)

            # Create figure
            fig, ax = plt.subplots(2, 5, figsize=(20, 8))
            fig.suptitle(f"Simulation {simulation_to_visualize} - Timestep {i}")

            for j in range(5):
                gt_index = i - 5 + j
                ax[0, j].imshow(sample_pressures[gt_index].cpu().numpy(), cmap='jet', aspect='auto')
                ax[0, j].set_title(f'Ground Truth {j} (t={i})')
                ax[0, j].axis("off")

                ax[1, j].imshow(predicted_pressure[0, j].cpu().numpy(), cmap='jet', aspect='auto')
                ax[1, j].set_title(f'Prediction {j} (t={i})')
                ax[1, j].axis("off")

                # print difference between ground truth and prediction
                diff = sample_pressures[j] - predicted_pressure[0, j]
                print(f"Difference between ground truth and prediction for timestep {i}: {diff}")

            # Save and log to WandB
            filename = os.path.join(output_dir, f"prediction_timestep_{i}.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)

            wandb.log({"Predictions": wandb.Image(filename, caption=f"Timestep {i}")})

            # Update `current_pressure` for recursive prediction
            current_pressure[:-1] = current_pressure[1:]
            current_pressure[-1] = predicted_pressure
            predicted_pressures.append(predicted_pressure[0, 0, :, :])
            if i > 50 + index_to_start:
                break






    

    #plot_recursive_predictions(clipped_sample_pressures, predicted_pressures, sample_times)





def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="fno_normalized_time_bundling_10_modes_lab_24_width_run")
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    # New model-specific arguments:
    parser.add_argument('--time_window', type=int, default=9, help="Number of channels for pressure input")
    parser.add_argument('--time_window_out', type=int, default=1, help="Number of channels for pressure output")
    parser.add_argument('--modes1', type=int, default=10)
    parser.add_argument('--modes2', type=int, default=10)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--cond_channels', type=int, default=31, help="Dimension of conditioning embedding (matches conditioning dimension[1])")
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dataset_path', type=str, default="/home/reid/projects/blast_waves/hdf5_dataset_1_simulation_per_file")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    wandb.init(project="blastformer_test", name=f"{args.run_name}_test")
    test(args)


if __name__ == '__main__':
    launch()