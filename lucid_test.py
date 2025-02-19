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
    seq_len = args.seq_len
    output_dim = 99
    input_dim = (99**2) // (patch_size**2)

    # Load model
    model = BlastFormer(input_dim, hidden_dim, num_layers, output_dim, patch_size).to(device)
    model_path = os.path.join("models", args.run_name, "best_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Loaded model from {model_path}")
    else:
        logging.error(f"Model checkpoint not found at {model_path}")
        return
    
    model.eval()  # Set model to evaluation mode

    l1 = nn.L1Loss()
    l2 = nn.MSELoss()

    total_l1_loss = 0
    total_l2_loss = 0

    # Run inference
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing")
        for batch in pbar:
            # Move inputs to device
            current_pressure = batch["source_pressure"].to(device)  # shape: (batch_size, 99, 99)
            next_pressures = batch["target_pressure"].to(device)  # Ground-truth target pressure
            charge_data = batch["source_charge_data"].to(device)
            wall_locations = batch["source_wall_locations"].to(device)
            current_time = batch["source_time"].to(device)

            # Model prediction
            predicted_pressure = model(current_pressure, charge_data, wall_locations, current_time).squeeze(1)

            # Compute loss
            l1_loss = l1(predicted_pressure, next_pressures)
            l2_loss = l2(predicted_pressure, next_pressures)

            total_l1_loss += l1_loss.item()
            total_l2_loss += l2_loss.item()

            pbar.set_postfix(L1_Loss=l1_loss.item(), MSE_Loss=l2_loss.item())

            # Log results to WandB
            wandb.log({
                "Batch L1 Loss": l1_loss.item(),
                "Batch MSE Loss": l2_loss.item(),
            })

    avg_l1_loss = total_l1_loss / len(dataloader)
    avg_l2_loss = total_l2_loss / len(dataloader)

    logging.info(f"Test completed. Avg L1 Loss: {avg_l1_loss:.6f}, Avg MSE Loss: {avg_l2_loss:.6f}")
    
    wandb.log({
        "Avg L1 Loss": avg_l1_loss,
        "Avg MSE Loss": avg_l2_loss
    })

    # Save and visualize predictions
    visualize_results(current_pressure, next_pressures, predicted_pressure, args.run_name)


def visualize_results(input_pressure, target_pressure, predicted_pressure, run_name):
    """Visualizes and saves pressure field comparisons."""
    num_samples = min(3, input_pressure.shape[0])  # Plot a few samples
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))

    for i in range(num_samples):
        axes[i, 0].imshow(input_pressure[i].cpu().numpy(), cmap="jet")
        axes[i, 0].set_title("Input Pressure")

        axes[i, 1].imshow(target_pressure[i].cpu().numpy(), cmap="jet")
        axes[i, 1].set_title("Target Pressure")

        axes[i, 2].imshow(predicted_pressure[i].cpu().numpy(), cmap="jet")
        axes[i, 2].set_title("Predicted Pressure")

    plt.tight_layout()
    results_path = os.path.join("results", run_name, "test_results.jpg")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    plt.savefig(results_path)
    wandb.log({"Test Results": wandb.Image(results_path)})
    logging.info(f"Saved test results to {results_path}")


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="lucid_blastformer_0")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=302)
    parser.add_argument('--dataset_path', type=str, default="/home/reid/projects/blast_waves/hdf5_dataset")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()
    
    wandb.init(project="blastformer_test", name=f"{args.run_name}_test")
    test(args)


if __name__ == '__main__':
    launch()
