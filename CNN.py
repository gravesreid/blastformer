import torch
import torch.nn as nn
import torch.nn.functional as F

class BlastCNN(nn.Module):
    def __init__(self,
                 base_dim=64,
                 dim_mults=(1, 2, 4, 8),
                 num_classes=1,
                 ):  
        super().__init__()
        self.conv1 = nn.Conv2d(4, base_dim*dim_mults[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_dim*dim_mults[0], base_dim*dim_mults[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_dim*dim_mults[1], base_dim*dim_mults[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(base_dim*dim_mults[2], base_dim*dim_mults[1], kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(base_dim*dim_mults[1], base_dim*dim_mults[0], kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(base_dim*dim_mults[0], 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)  # No activation if output is regression
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        return x
