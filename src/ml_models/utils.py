from collections import OrderedDict

import torch
import torch.nn as nn
from flwr.common import NDArrays

from src.scripts.helper import metadata


def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def to_onehot(labels, num_classes, device):
    onehot = torch.zeros(labels.size(0), num_classes, device=device)
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def create_synthetic_data(
    model, label, device, samples_per_class, batch_size, mode
) -> NDArrays:
    model.eval()
    model.to(device)

    # Create synthetic data in memory (or in batches if memory is constrained)
    synthetic_data = []
    synthetic_labels = []

    # Generate in batches to be memory efficient
    for start_idx in range(0, samples_per_class, batch_size):
        current_batch_size = min(batch_size, samples_per_class - start_idx)

        latent_dim = 0
        images = []

        if mode == "HFedCVAE":
            latent_dim = metadata["HFedCVAE"]["vae_parameters"]["latent_dim"]
        elif mode == "HFedCGAN":
            latent_dim = metadata["HFedCGAN"]["generator_parameters"]["latent_dim"]
        elif mode == "HFedCVAEGAN":
            latent_dim = metadata["HFedCVAEGAN"]["vae_parameters"]["latent_dim"]

        # Sample z
        z = torch.randn(current_batch_size, latent_dim, device=device)

        if mode == "HFedCVAE" or mode == "HFedCVAEGAN":
            with torch.no_grad():
                expanded = model.fc_expand(z)
                expanded = expanded.view(current_batch_size, *model.enc_shape)
                images = model.decoder(expanded)
        elif mode == "HFedCGAN":
            with torch.no_grad():
                images = model(z)

        synthetic_data.append(images.cpu())
        synthetic_labels.append(
            torch.tensor(
                [label for _ in range(current_batch_size)],
                dtype=torch.long,
            ).cpu()
        )

    # Concatenate all batches
    synthetic_data = torch.cat(synthetic_data, dim=0)
    synthetic_labels = torch.cat(synthetic_labels, dim=0)

    return [synthetic_data.numpy(), synthetic_labels.numpy()]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def vae_loss(recon_x, x, mu, logvar):
    """
    recon_x: reconstructed batch  [B, C, H, W]
    x:       original batch        [B, C, H, W]
    mu:      mean of q(z|x,y)
    logvar:  log variance of q(z|x,y)
    """

    # BCE reconstruction loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")

    # KL divergence term
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + 0.3 * kl_loss, recon_loss, kl_loss
