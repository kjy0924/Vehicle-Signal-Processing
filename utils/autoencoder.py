import torch
import torch.nn as nn


class PSDAutoencoder(nn.Module):
    def __init__(self, input_dim=129, latent_dim=4):
        super(PSDAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    @torch.no_grad()
    def compute_anomaly_score(self, psd_tensor):
        self.eval()

        reconstructed = self(psd_tensor)

        mse_loss = nn.MSELoss(reduction='none')(reconstructed, psd_tensor).mean().item()

        return mse_loss, reconstructed.squeeze().numpy()