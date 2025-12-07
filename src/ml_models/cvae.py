import torch
import torch.nn as nn

from src.ml_models.decoder import Decoder
from src.ml_models.encoder import Encoder
from src.scripts.helper import metadata


class CVAE(nn.Module):
    def __init__(
        self,
        num_classes=metadata["num_classes"],
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
        # Input to FC now includes condition y
        self.fc_mu = nn.Linear(self.flatten_dim + num_classes, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim + num_classes, latent_dim)

        # ----- Decoder -----
        # Decoder input: z + y
        self.fc_expand = nn.Linear(latent_dim + num_classes, self.flatten_dim)

        self.decoder = Decoder(C_enc, h_dim, n_res_layers, res_h_dim, input_shape[0])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        """
        x : input image batch
        y : conditioning batch (one-hot or dense), shape [B, num_classes]
        """
        h = self.encoder(x)
        B = h.size(0)

        h = h.flatten(start_dim=1)

        h_cond = torch.cat([h, y], dim=1)

        mu = self.fc_mu(h_cond)
        logvar = self.fc_logvar(h_cond)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        z_cond = torch.cat([z, y], dim=1)

        z_expanded = self.fc_expand(z_cond).view(B, *self.enc_shape)

        x_recon = self.decoder(z_expanded)

        return x_recon, mu, logvar
