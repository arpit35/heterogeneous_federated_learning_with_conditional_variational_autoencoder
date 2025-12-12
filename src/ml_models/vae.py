import torch
import torch.nn as nn

from src.ml_models.decoder import Decoder
from src.ml_models.encoder import Encoder
from src.scripts.helper import metadata


class vae(nn.Module):
    def __init__(
        self,
        h_dim=metadata["h_dim"],
        res_h_dim=metadata["res_h_dim"],
        n_res_layers=metadata["n_res_layers"],
        latent_dim=metadata["latent_dim"],
        input_shape=(
            metadata["num_channels"],
            metadata["image_width"],
            metadata["image_height"],
        ),
    ):
        super().__init__()

        # ----- Encoder -----
        self.encoder = Encoder(input_shape[0], h_dim, n_res_layers, res_h_dim)

        # Compute shape after encoder
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            enc_out = self.encoder(dummy)
            _, C_enc, H_enc, W_enc = enc_out.shape
            self.enc_shape = (C_enc, H_enc, W_enc)
            self.flatten_dim = C_enc * H_enc * W_enc

        # ----- Latent space -----
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # ----- Decoder -----
        self.fc_expand = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = Decoder(C_enc, h_dim, n_res_layers, res_h_dim, input_shape[0])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        x : input image batch
        """
        h = self.encoder(x)
        B = h.size(0)

        h = h.flatten(start_dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        z = self.fc_expand(z).view(B, *self.enc_shape)

        x_recon = self.decoder(z)

        return x_recon, mu, logvar
