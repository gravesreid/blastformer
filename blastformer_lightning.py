import os
from torch import optim, nn, utils, Tensor
import torch
from torchvision.transforms import ToTensor
import lightning as L
from utils import CFDFeatureEmbedder, patchify_batch, unpatchify_batch, custom_collate
from dataset import BlastDataset
from blastformer_transformer import BlastFormer

# hyperparameters for blastformer
input_dim = 81
hidden_dim = 256
output_dim = 81
seq_len = 302
patch_size = 11
num_layers = 4

blastformer = BlastFormer(input_dim, hidden_dim, num_layers, output_dim, seq_len, patch_size)

class LightningBlastFormer(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0)

        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        current_pressure = batch["pressure"][:, :1, :]
        next_pressures = batch["pressure"][:, -1, :]
        next_patches = patchify_batch(next_pressures, patch_size)
        charge_data = batch["charge_data"]
        wall_locations = batch["wall_locations"][:, 0, :]
        current_time = batch["time"][:, :1, :]
        next_time = batch["time"][:, 1:, :]
        outputs = self.model(current_pressure, charge_data, wall_locations, current_time)
        predicted_pressure = outputs[:, :next_patches.shape[1], :]
        loss = self.criterion(predicted_pressure, next_patches)
        log_values = {'train_loss': loss, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(log_values, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer
    
    def on_train_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss)
        self.scheduler.step()


# Load Dataset and Dataloader
root_dir = "/home/reid/projects/blast_waves/dataset_parallel_processed_large/train"
dataset = BlastDataset(root_dir)
dataloader = utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=10, collate_fn=custom_collate)

blastformer = LightningBlastFormer(blastformer)

trainer = L.Trainer(max_epochs=20)
trainer.fit(blastformer, dataloader)