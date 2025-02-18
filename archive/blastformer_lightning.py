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
        source_pressure = batch["source_pressure"]
        target_pressure = batch["target_pressure"]
        next_patches = patchify_batch(target_pressure, self.patch_size)
        source_time = batch["source_time"]
        target_time = batch["target_time"]
        source_wall_locations = batch["source_wall_locations"]
        target_wall_locations = batch["target_wall_locations"]
        source_charge_data = batch["source_charge_data"]
        target_charge_data = batch["target_charge_data"]
        outputs = self.model(source_pressure, source_charge_data, source_wall_locations, source_time)
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
    batch_size=128,
    shuffle=True,
    num_workers=min(8, os.cpu_count() - 1),  # Multi-worker loading
    )

    blastformer = LightningBlastFormer(blastformer, patch_size=patch_size)

    trainer = L.Trainer(max_epochs=20)
    trainer.fit(blastformer, dataloader)

if __name__ == "__main__":
    main()