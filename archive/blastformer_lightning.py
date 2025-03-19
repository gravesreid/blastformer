import os
from torch import optim, nn, utils, Tensor
import torch
from torchvision.transforms import ToTensor
import lightning as L
from hdf5_dataset_new import BlastDataset
from oformer.oformer import OFormerModule
import wandb


    
def main():
    wandb.init(project='blastOformer', name='blastOformer')

    blastformer = OFormerModule(
        modelconfig={
            "encoder": {
                "input_channels": 1,
                "in_emb_dim": 64,
                "out_seq_emb_dim": 32,
                "heads": 4,
                "depth": 4,
                "res": 32,
        },
        "decoder": {
            "latent_channels": 128,
            "out_channels": 1,
    },
        },
        trainconfig={
            "dist": "normal",
            "learning_rate": 1e-4,
            "batch_size": 8,
            "max_epochs": 20,
            "scheduler": "Cosine",
            "dataset_size": 100,
        },
    )
    # Load Dataset and Dataloader
    root_dir = "/home/reid/projects/blast_waves/hdf5_dataset_ultra_low_res_1_simulation_per_file"
    dataset = BlastDataset(root_dir)
    dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=min(8, os.cpu_count() - 1),  # Multi-worker loading
    )


    trainer = L.Trainer(max_epochs=20)
    trainer.fit(blastformer, dataloader)


if __name__ == "__main__":
    main()