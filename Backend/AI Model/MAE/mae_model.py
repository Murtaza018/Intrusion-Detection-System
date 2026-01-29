import torch
import torch.nn as nn
import torch.nn.functional as F

class MAEModel(nn.Module):
    def __init__(self, input_dim=78, grid_size=9):
        super(MAEModel, self).__init__()
        self.grid_size = grid_size
        self.total_pixels = grid_size * grid_size # 81
        self.input_dim = input_dim
        
        # Encoder: Learning the "Visual" patterns
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 4x4
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64)
        )
        
        # Decoder: Reconstructing the masked image
        self.decoder = nn.Sequential(
            nn.Linear(64, 32 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=1, padding=0), # 9x9
            nn.Sigmoid()
        )

    def forward(self, x, mask_ratio=0.5):
        # 1. Padding 78 -> 81
        batch_size = x.shape[0]
        padding = torch.zeros((batch_size, self.total_pixels - self.input_dim)).to(x.device)
        x_padded = torch.cat([x, padding], dim=1)
        
        # 2. Reshape to 9x9 Image
        x_img = x_padded.view(-1, 1, self.grid_size, self.grid_size)
        
        # 3. Apply Mask (Only during training or for anomaly scoring)
        if self.training or mask_ratio > 0:
            mask = torch.rand(x_img.shape).to(x.device) > mask_ratio
            x_masked = x_img * mask
        else:
            x_masked = x_img
            
        # 4. Autoencode
        latent = self.encoder(x_masked)
        reconstruction = self.decoder(latent)
        
        return reconstruction, x_img