import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ml_models.decoder import Decoder
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


class HFedCVAE(nn.Module):
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

        self.mu = nn.Linear(fc2_out_features, metadata["decoder_latent_dim"])
        self.logvar = nn.Linear(fc2_out_features, metadata["decoder_latent_dim"])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def combined_loss(
        self,
        images,
        labels,
        logits,
        recon_x,
        mu,
        logvar,
    ):
        # Classification loss
        classification_loss = F.cross_entropy(logits, labels, reduction="sum")

        if recon_x is not None:
            width = metadata["image_width"]
            height = metadata["image_height"]
            # recon_x and x in [0,1]
            # Reconstruction loss (binary cross entropy) summed over pixels
            BCE = F.binary_cross_entropy(
                recon_x.view(-1, width * height),
                images.view(-1, width * height),
                reduction="sum",
            )
            # KL divergence between q(z|x) and N(0,1)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            return classification_loss + BCE + KLD, classification_loss, BCE, KLD
        else:
            return classification_loss, classification_loss, None, None

    def forward(self, x, y, detach_decoder=False):
        x = F.relu(self.conv1(x))  # → 16x28x28
        x = F.relu(self.conv2(x))  # → 32x24x24
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))  # → 2000
        x = F.relu(self.fc2(x))  # → 500

        logits = self.fc3(x)  # → num_classes

        if detach_decoder:
            # Skip decoder forward pass and return None for recon_x
            recon_x = None
            mu = None
            logvar = None
        else:
            mu = self.mu(x)
            logvar = self.logvar(x)
            z = self.reparameterize(mu, logvar)
            # Reconstruction
            recon_x = self.decoder(z, y)

        return logits, recon_x, mu, logvar
