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
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import matplotlib.pyplot as plt
from utils import *
from hdf5_dataset_max_pressure import *
from BlastOFormer import BlastOFormer


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def train(args):
    setup_logging(args.run_name)
    device = args.device

    training_dataset = BlastDataset(args.dataset_path, split="train", normalize=True)
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=min(16, os.cpu_count() - 1))
    l = len(training_dataloader)

    validation_dataset = BlastDataset(args.dataset_path, split="val", normalize=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=min(16, os.cpu_count() - 1))
    if len(validation_dataloader) == 0:
        logging.error("Validation dataloader is empty. Check the dataset path.")
        return
    
    model = BlastOFormer(
        encoder_input_channels=4,
        encoder_in_emb_dim=256,
        encoder_out_seq_emb_dim=256,
        encoder_heads=8,
        encoder_depth=4,
        encoder_res=99,
        decoder_latent_channels=256,
        decoder_out_channels=1,
        decoder_res=99,
        decoder_scale=16,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    wandb.init(project="BlastOFormer", name=args.run_name, config=args)
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
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        pbar = tqdm(training_dataloader, desc="Training", total=l)
        model.train()
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            max_pressure = batch["max_pressure"].to(device)
            max_pressure = rearrange(max_pressure, "b w h -> b (w h)").unsqueeze(-1)
            probe_positions = batch["probe_positions"][:,:,:,0:2].to(device)
            probe_positions = rearrange(probe_positions,"b w h c -> b (w h) c")
            x = batch["input_tensor"].to(device)
            x = rearrange(x,"b w h c -> b (w h) c")
            y = max_pressure
            prediction = model(x, probe_positions)
            loss = l2(prediction, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"Loss": loss.item()})
            training_loss.append(loss.item())
            wandb.log({"train/loss": loss.item()})
            logger.add_scalar("Loss/train", loss.item(), epoch * l + i)

        scheduler.step()


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/reid/projects/blast_waves/hdf5_dataset_max_pressure")
    parser.add_argument("--run_name", type=str, default="BlastOFormer_Lab")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    launch()
