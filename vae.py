
import torch
import torch.nn as nn


# =========================
# Encoder
# =========================

class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU()
        )

        self.fc_mean = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


# =========================
# Decoder
# =========================

class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)

        return self.deconv_layers(x)


# =========================
# Complete VAE
# =========================

class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)

        return mean + epsilon * std

    def forward(self, x):
        mean, logvar = self.encoder(x)

        z = self.reparameterize(mean, logvar)

        reconstructed = self.decoder(z)

        return reconstructed, mean, logvar


# =========================
# VAE Loss
# =========================

def vae_loss(reconstructed, original, mean, logvar, beta=1.0):

    recon_loss = (
        (reconstructed - original).pow(2)
        .flatten(start_dim=1)
        .sum(dim=1)
        .mean()
    )

    kl_loss = -0.5 * (
        1 + logvar - mean.pow(2) - logvar.exp()
    ).sum(dim=1).mean()

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
