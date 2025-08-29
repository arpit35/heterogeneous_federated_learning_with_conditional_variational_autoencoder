import torch
import torch.nn as nn

from src.scripts.helper import metadata

# Output size calculation: (input_size - 1) × stride - 2 × padding + kernel_size + output_padding = input_size*2 + output_padding


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.width_dim = int(metadata["image_width"] / 4)
        self.height_dim = int(metadata["image_height"] / 4)

        self.fc = nn.Linear(
            metadata["decoder_latent_dim"] + metadata["num_classes"],
            64 * self.width_dim * self.height_dim,
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32,
                metadata["num_channels"],
                4,
                2,
                1,
                metadata["image_width"] - (self.width_dim * 4),
            ),
            nn.Sigmoid(),
        )

    def forward(self, z, y):
        z = self.fc(torch.cat([z, y], dim=1))
        z = z.view(-1, 64, self.width_dim, self.height_dim)
        return self.net(z)


class DecoderLatentSpace(nn.Module):
    def __init__(self, in_latent_dim):
        super().__init__()

        self.mu = nn.Linear(in_latent_dim, metadata["decoder_latent_dim"])
        self.logvar = nn.Linear(in_latent_dim, metadata["decoder_latent_dim"])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)
        return self.reparameterize(mu, logvar), mu, logvar
