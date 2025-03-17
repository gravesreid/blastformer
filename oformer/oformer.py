import torch
import torch.nn.functional as F
import lightning as L
from oformer.decoder_module import PointWiseDecoder2DSimple
from oformer.encoder_module import SpatialEncoder2D
import wandb

######################
# Module
######################

class OFormerModule(L.LightningModule):
    def __init__(self,
                 modelconfig,
                 trainconfig,
                 normalizer=None,
                 batch_size=16,
                 accumulation_steps=1,
                 ckpt_path=None,
                 cond_dim = 10
                 ):
        super().__init__()

        # add mlp to process conditining
        self.cond_mlp = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, cond_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(cond_dim, cond_dim)
        )

        self.encoder = SpatialEncoder2D(**modelconfig["encoder"])
        self.decoder = PointWiseDecoder2DSimple(**modelconfig["decoder"])
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.trainconfig = trainconfig

        self.dist = trainconfig['dist']

        self.save_hyperparameters()

        print("Training with batch size", self.batch_size)
        print("Training with accumulation steps", self.accumulation_steps)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, x, pos, cond=None):
        if cond is not None:
            # Process conditional info
            cond_emb = self.cond_mlp(cond)  # [batch, cond_dim]
            # Expand and concatenate to x along the feature dimension:
            cond_expanded = cond_emb.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, cond_expanded], dim=-1)

        z = self.encoder(x, pos)
        pred = self.decoder(z, pos, pos)
        return pred  
    

    def training_step(self, batch, batch_idx):
        charge_center = batch["charge_center"]
        charge_mass = batch["charge_mass"]
        wall_1 = batch["wall_1"]
        wall_2 = batch["wall_2"]
        wall_3 = batch["wall_3"]
        times = batch["times"]
        cond = torch.cat((charge_center, charge_mass.unsqueeze(1), wall_1, wall_2, wall_3), dim=1)

        pressures = batch["pressures"]
        print(f'pressures shape: {pressures.shape}')
        pos = batch["probe_positions"]
        pos = pos[ :, :, :2] # 1 m 2
        
        #inputs = self.normalizer.normalize(inputs)  # normalize inputs to [-1, 1]
        rollout_length = 20
        start_time = torch.randint(low=0, high=pressures.shape[1] - rollout_length, size=(1,)).item()

        rolout_predictions = []
        x = pressures[:, start_time]
        current_state = x
        for i in range(rollout_length):
            print(f'current_state shape: {current_state.shape}')
            print(f'pos shape: {pos.shape}')
            pred = self(current_state, pos)
            rolout_predictions.append(pred)
            current_state = pred
        rollout_pred = torch.stack(rolout_predictions, dim=1)
        target = pressures[:, start_time + 1:start_time + 1 + rollout_length]

        loss = F.mse_loss(rollout_pred, target) # 1 m c, 1 m c

        self.log("train/mse_loss", loss, prog_bar=False,
                      logger=True, on_step=True, on_epoch=True,
                      sync_dist=self.dist,)
        wandb.log({"train/mse_loss": loss})
        
        return loss 

    def validation_step(self, batch, batch_idx, eval=False):
        charge_center = batch["charge_center"]
        charge_mass = batch["charge_mass"]
        wall_1 = batch["wall_1"]
        wall_2 = batch["wall_2"]
        wall_3 = batch["wall_3"]
        times = batch["times"]
        cond = torch.cat((charge_center, charge_mass.unsqueeze(1), wall_1, wall_2, wall_3), dim=1)

        pressures = batch["pressures"]
        pos = batch["probe_positions"]
        pos = pos[:, :, :2]  # Extract only the x,y positions

        # Directly predict the last timestep, similar to training
        pred = self(pressures, pos)  # Model predicts based on full sequence
        target = pressures[:, -1]  # Last timestep as the target

        loss = F.mse_loss(pred, target)  # Compute MSE loss

        # Denormalized L1 loss (optional, depends on training approach)
        pred_denorm = self.normalizer.denormalize(pred)
        target_denorm = self.normalizer.denormalize(target)
        loss_denorm = F.l1_loss(pred_denorm, target_denorm)

        # Compute rollout loss if needed
        if eval:
            return loss_denorm.item(), pred  # Return full sequence predictions if evaluating

        self.log("val/mse_loss", loss, prog_bar=False,
                logger=True, on_step=False, on_epoch=True,
                sync_dist=self.dist)
        self.log("val/l1_loss", loss_denorm, prog_bar=False,
                logger=True, on_step=False, on_epoch=True,
                sync_dist=self.dist)

        return loss


    def configure_optimizers(self):
        lr = self.trainconfig["learning_rate"]
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                    list(self.decoder.parameters()),
                                    lr=lr,)
            
        effective_batch_size = self.batch_size * self.accumulation_steps
        if self.trainconfig["scheduler"] == "OneCycle":
            scheduler_ae = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt_ae,
                                                            max_lr=lr,
                                                            total_steps=self.trainconfig["max_epochs"] * (self.trainconfig["dataset_size"] // effective_batch_size  + 1),
                                                            pct_start=self.trainconfig["pct_start"],)

        elif self.trainconfig["scheduler"] == "Cosine":
            scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt_ae,
                                                                      T_max=self.trainconfig["max_epochs"] * (self.trainconfig["dataset_size"] // effective_batch_size  + 1),)
        else:
            scheduler_ae = None
        return [opt_ae], [scheduler_ae]