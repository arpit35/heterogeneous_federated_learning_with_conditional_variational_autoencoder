import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ml_models.decoder import Decoder
from src.ml_models.generic_encoder import GenericEncoder
from src.ml_models.personalized_encoder import PersonalizedEncoder
from src.scripts.helper import metadata

cnn_config = {
    "cnn1": {
        "conv1_filters": 16,
        "conv2_filters": 32,
        "conv1_kernel_size": 5,
        "conv2_kernel_size": 5,
        "fc1_out_features": 2000,
        "fc2_out_features": 500,
    },
    "cnn2": {
        "conv1_filters": 16,
        "conv2_filters": 16,
        "conv1_kernel_size": 5,
        "conv2_kernel_size": 5,
        "fc1_out_features": 2000,
        "fc2_out_features": 500,
    },
    "cnn3": {
        "conv1_filters": 16,
        "conv2_filters": 32,
        "conv1_kernel_size": 5,
        "conv2_kernel_size": 5,
        "fc1_out_features": 1000,
        "fc2_out_features": 500,
    },
    "cnn4": {
        "conv1_filters": 16,
        "conv2_filters": 32,
        "conv1_kernel_size": 5,
        "conv2_kernel_size": 5,
        "fc1_out_features": 800,
        "fc2_out_features": 500,
    },
    "cnn5": {
        "conv1_filters": 16,
        "conv2_filters": 32,
        "conv1_kernel_size": 5,
        "conv2_kernel_size": 5,
        "fc1_out_features": 500,
        "fc2_out_features": 500,
    },
}


class HFedPVA(nn.Module):
    def __init__(self, cnn_type: str):
        super().__init__()
        (
            conv1_filters,
            conv2_filters,
            conv1_kernel_size,
            conv2_kernel_size,
            fc1_out_features,
            fc2_out_features,
        ) = cnn_config[cnn_type].values()

        self.generic_encoder = GenericEncoder()
        self.personalized_encoder = PersonalizedEncoder()
        self.decoder = Decoder()

        self.cnn_encoder_connector = nn.Linear(
            in_features=fc1_out_features,
            out_features=metadata["encoder_input_dim"],
        )

        # Input: 1x32x32
        self.conv1 = nn.Conv2d(
            in_channels=metadata["num_channels"],
            out_channels=conv1_filters,
            kernel_size=conv1_kernel_size,
        )  # → 16x28x28
        self.conv2 = nn.Conv2d(
            in_channels=conv1_filters,
            out_channels=conv2_filters,
            kernel_size=conv2_kernel_size,
        )  # → 32x24x24

        # Flatten size: 32 × 24 × 24 = 18432
        self.fc1 = nn.Linear(
            conv2_filters
            * (metadata["image_height"] - conv1_kernel_size + 1 - conv2_kernel_size + 1)
            * (metadata["image_width"] - conv1_kernel_size + 1 - conv2_kernel_size + 1),
            fc1_out_features,
        )
        self.fc2 = nn.Linear(fc1_out_features, fc2_out_features)
        self.fc3 = nn.Linear(
            fc2_out_features, metadata["num_classes"]
        )  # Output: num_classes

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def combined_loss(
        self,
        y_logits,
        y_true,
        recon_x,
        x,
        mu_z,
        logvar_z,
        mu_c,
        logvar_c,
        all_mu_c,
        all_logvar_c,
    ):
        x_flat = x.view(x.size(0), -1)
        # Classification loss
        ce_loss = F.cross_entropy(y_logits, y_true, reduction="sum")

        # Reconstruction loss
        BCE = F.binary_cross_entropy(recon_x, x_flat, reduction="sum")
        BCE = torch.clamp(BCE, min=-25000, max=25000)

        # KL Regularizers (Eq.6-7)
        # R_z: KL(q(z|x) || p(z)) (p(z) = N(0,I))
        KLD_z = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

        # Compute p_k(c) as mixture distribution (Eq.5)
        mu_c_mix = torch.mean(all_mu_c, dim=0)
        logvar_c_mix = torch.mean(all_logvar_c, dim=0)

        # KL(q(c|x,z) || p(c)) (p(c) = N(0,I))
        KLD_c_prior = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())

        # KL(q(c|x,z) || p_k(c))
        var_c = logvar_c.exp()
        var_c_mix = logvar_c_mix.exp()
        KLD_c_client = 0.5 * torch.sum(
            logvar_c_mix - logvar_c + (var_c + (mu_c - mu_c_mix).pow(2)) / var_c_mix - 1
        )

        # Slack regularizer (Eq.5,7)
        R_c = torch.max(0.1 + KLD_c_client, KLD_c_prior)

        # Final loss (Eq.4)
        feddva_loss = BCE + KLD_z + R_c

        total_loss = ce_loss + feddva_loss

        return total_loss, ce_loss, BCE, KLD_z, R_c

    def forward(self, x):
        x = F.relu(self.conv1(x))  # → 16x28x28
        x = F.relu(self.conv2(x))  # → 32x24x24
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))  # → 2000
        connector = self.cnn_encoder_connector(x)  # → encoder_input_dim
        x = F.relu(self.fc2(x))  # → 500
        x = self.fc3(x)  # → num_classes

        # Generic representation
        mu_z, logvar_z = self.generic_encoder(connector)
        z = self.reparameterize(mu_z, logvar_z)

        # Personalized representation
        mu_c, logvar_c = self.personalized_encoder(connector, z)
        c = self.reparameterize(mu_c, logvar_c)

        # Reconstruction
        recon_x = self.decoder(z, c)

        return x, recon_x, mu_z, logvar_z, mu_c, logvar_c
