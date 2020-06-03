import torch
import torch.nn as nn
import torch.nn.functional as F


dimZ = 100
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(image_h * image_w * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, dimZ),
            nn.BatchNorm1d(dimZ),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(dimZ , 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, image_h * image_w * 3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
      
        latent_code = self.encoder(x)
        reconstruction = self.decoder(latent_code)
        
        return reconstruction, latent_code
