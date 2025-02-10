import os
from torch import optim, nn, utils, Tensor
import torch
from torchvision.transforms import ToTensor
import lightning as L
from utils import CFDFeatureEmbedder, patchify_batch, unpatchify_batch, custom_collate
from hdf5_dataset import BlastDataset
from blastformer_transformer import BlastFormer


class LightningBlastFormer(L.LightningModule):
    def __init__(self, model, patch_size=11):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0)
        self.save_hyperparameters()
        self.patch_size = patch_size

        
    def forward(self, current_pressure, charge_data, wall_locations, current_time):
        return self.model(current_pressure, charge_data, wall_locations, current_time)
    
    def training_step(self, batch, batch_idx):
        current_pressure = batch[0]["pressure"]
        next_pressures = batch[0]["pressure"]
        next_patches = patchify_batch(next_pressures, self.patch_size)
        charge_data = batch[0]["charge_data"]
        wall_locations = batch[0]["wall_locations"]
        current_time = batch[0]["time"]
        next_time = batch[1]["time"]
        outputs = self.model(current_pressure, charge_data, wall_locations, current_time)
        predicted_pressure = outputs[:, :next_patches.shape[1], :]
        loss = self.criterion(predicted_pressure, next_patches)
        log_values = {'train_loss': loss, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(log_values, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'lr_scheduler' : {
                'scheduler': self.scheduler,
                'interval': 'epoch'
            }
        }
    


def main():
    # hyperparameters for blastformer
    input_dim = 81
    hidden_dim = 256
    output_dim = 81
    seq_len = 302
    patch_size = 11
    num_layers = 4

    blastformer = BlastFormer(input_dim, hidden_dim, num_layers, output_dim, seq_len, patch_size)
    # Load Dataset and Dataloader
    root_dir = "/home/reid/projects/blast_waves/hdf5_dataset"
    dataset = BlastDataset(root_dir)
    dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=min(12, os.cpu_count() - 1),  # Multi-worker loading
    pin_memory=True,  # If using GPU
    prefetch_factor=4,  # Reduce CPU bottleneck
    persistent_workers=True  # Avoid restarting workers
    )

    blastformer = LightningBlastFormer(blastformer, patch_size=patch_size)

    trainer = L.Trainer(max_epochs=20)
    trainer.fit(blastformer, dataloader)

if __name__ == "__main__":
    main()