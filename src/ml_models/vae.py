import torch
import torch.nn as nn

from src.ml_models.decoder import Decoder
from src.ml_models.encoder import Encoder
from src.ml_models.utils import one_hot_labels
from src.scripts.helper import metadata


class CVAE(nn.Module):
    """Simplified CVAE that only concatenates labels at the input and latent space."""

    def __init__(
        self,
        h_dim,
        res_h_dim,
        n_res_layers,
        latent_dim,
        num_classes=metadata["num_classes"],
        input_shape=(
            metadata["num_channels"],
            metadata["image_width"],
            metadata["image_height"],
        ),
    ):
        super().__init__()

        self.num_classes = num_classes

        # ----- Encoder -----
        # Input channels + one-hot labels
        encoder_input_channels = input_shape[0] + num_classes
        self.encoder = Encoder(encoder_input_channels, h_dim, n_res_layers, res_h_dim)

        # Compute shape after encoder
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy_labels = torch.zeros(1, num_classes, *input_shape[1:])
            enc_input = torch.cat([dummy, dummy_labels], dim=1)
            enc_out = self.encoder(enc_input)
            _, C_enc, H_enc, W_enc = enc_out.shape
            self.enc_shape = (C_enc, H_enc, W_enc)
            self.flatten_dim = C_enc * H_enc * W_enc

        # ----- Latent space -----
        self.fc_mu = nn.Linear(self.flatten_dim + num_classes, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim + num_classes, latent_dim)

        # ----- Decoder -----
        self.fc_expand = nn.Linear(latent_dim + num_classes, self.flatten_dim)

        # Decoder
        decoder_input_channels = C_enc + num_classes
        self.decoder = Decoder(
            decoder_input_channels, h_dim, n_res_layers, res_h_dim, input_shape[0]
        )

    def vae_loss(self, recon_x, x, mu, logvar, epoch):
        """
        recon_x: reconstructed batch  [B, C, H, W]
        x:       original batch        [B, C, H, W]
        mu:      mean of q(z|x,y)
        logvar:  log variance of q(z|x,y)
        """
        beta = min(1, 1 * epoch / 5)

        # BCE reconstruction loss
        recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")

        # KL divergence term
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + beta * kl_loss, recon_loss, kl_loss

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        batch_size = x.size(0)

        # One-hot encode labels
        encoder_one_hot_spatial = one_hot_labels(
            labels, self.num_classes, (x.shape[2], x.shape[3])
        )

        # Encoder
        x_cond = torch.cat([x, encoder_one_hot_spatial], dim=1)
        h = self.encoder(x_cond)
        h = h.flatten(start_dim=1)

        # Concatenate with one-hot for latent space
        one_hot = one_hot_labels(labels, self.num_classes)
        h_with_label = torch.cat([h, one_hot], dim=1)

        mu = self.fc_mu(h_with_label)
        logvar = self.fc_logvar(h_with_label)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decoder
        z_with_label = torch.cat([z, one_hot], dim=1)
        z_expanded = self.fc_expand(z_with_label).view(batch_size, *self.enc_shape)

        # Concatenate with spatial one-hot
        decoder_one_hot_spatial = one_hot_labels(
            labels, self.num_classes, (self.enc_shape[1], self.enc_shape[2])
        )
        z_cond = torch.cat([z_expanded, decoder_one_hot_spatial], dim=1)
        x_recon = self.decoder(z_cond)

        return x_recon, mu, logvar
