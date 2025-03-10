import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import wandb

# Import the UNetAutoencoder model
from Unet import UNetAutoencoder

class TrainUNetAutoencoder:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the model
        self.model = UNetAutoencoder(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            features=config["features"],
            bilinear=config["bilinear"]
        ).to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        
        # Set up learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.75, 
            patience=5, 
            verbose=True
        )
        
        # Initialize wandb
        if config["use_wandb"]:
            wandb.init(project=config["wandb_project"], name=config["wandb_run_name"], config=config)
        
        # Create directories for saving models and results
        os.makedirs(config["model_save_dir"], exist_ok=True)
        os.makedirs(config["results_dir"], exist_ok=True)
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            pressure = batch["pressures"][:, 0, :, :].to(self.device).unsqueeze(1)

        
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(pressure)
            
            # Calculate loss
            loss = self.criterion(outputs, pressure)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                inputs = batch["pressures"][:,0,:,:].to(self.device).unsqueeze(1)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, inputs)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def save_model(self, epoch, loss):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(self.config["model_save_dir"], f"model_epoch_{epoch}.pt"))
    
    def visualize_results(self, inputs, outputs, epoch):
        """Visualize and save reconstruction results"""
        # Convert tensors to numpy arrays
        inputs_np = inputs.cpu().numpy()
        outputs_np = outputs.cpu().numpy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Plot original and reconstructed images
        for i in range(4):
            if i < inputs_np.shape[0]:
                # Original
                axes[0, i].imshow(inputs_np[i, 0], cmap='viridis')
                axes[0, i].set_title(f"Original {i+1}")
                axes[0, i].axis('off')
                
                # Reconstructed
                axes[1, i].imshow(outputs_np[i, 0], cmap='viridis')
                axes[1, i].set_title(f"Reconstructed {i+1}")
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["results_dir"], f"reconstruction_epoch_{epoch}.png"))
        plt.close()
        
        # Log to wandb if enabled
        if self.config["use_wandb"]:
            wandb.log({
                "reconstructions": wandb.Image(
                    os.path.join(self.config["results_dir"], f"reconstruction_epoch_{epoch}.png")
                )
            })
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config["epochs"]):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.config['epochs']}, "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}")
            
            # Log to wandb
            if self.config["use_wandb"]:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch + 1, val_loss)
                print(f"Saved model with improved validation loss: {val_loss:.6f}")
            
            # Visualize results periodically
            if (epoch + 1) % self.config["visualize_every"] == 0:
                # Get a batch of validation data
                batch = next(iter(val_loader))
                inputs = batch["pressures"][:, 0, :, :].to(self.device).unsqueeze(1)
                
                # Generate reconstructions
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(inputs)
                
                # Visualize
                self.visualize_results(inputs, outputs, epoch + 1)
        
        print("Training completed!")
        
        # Close wandb
        if self.config["use_wandb"]:
            wandb.finish()

# Example usage with your BlastDataset
def main():
    # Define configuration
    config = {
        # Model parameters
        "in_channels": 1,
        "out_channels": 1,
        "features": [64, 128, 256, 512],
        "bilinear": False,
        
        # Training parameters
        "batch_size": 128,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 100,
        "visualize_every": 5,
        
        # Paths
        "model_save_dir": "models/unet_autoencoder",
        "results_dir": "results/unet_autoencoder",
        
        # Wandb
        "use_wandb": True,
        "wandb_project": "unet_autoencoder",
        "wandb_run_name": "blast_waves_reconstruction"
    }
    
    # Import your dataset
    from hdf5_dataset_new import BlastDataset
    
    # Create dataset with single channel pressure fields
    train_dataset = BlastDataset(
        root_dir="/home/reid/projects/blast_waves/hdf5_dataset_1_simulation_per_file",
        split="train",
        normalize=True
    )
    
    val_dataset = BlastDataset(
        root_dir="/home/reid/projects/blast_waves/hdf5_dataset_1_simulation_per_file",
        split="val",
        normalize=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=max(4, os.cpu_count() - 2)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=max(4, os.cpu_count() - 2)
    )
    
    # Initialize trainer and start training
    trainer = TrainUNetAutoencoder(config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()