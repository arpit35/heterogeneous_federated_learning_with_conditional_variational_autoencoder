import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from flwr.common import NDArrays

from src.scripts.helper import metadata


def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def one_hot_labels(labels, num_classes, spatial_size=None):
    """Convert labels to one-hot and optionally make spatial."""
    one_hot = F.one_hot(labels, num_classes).float()

    if spatial_size is not None:
        H, W = spatial_size
        one_hot = one_hot.unsqueeze(-1).unsqueeze(-1)  # [B, num_classes, 1, 1]
        one_hot = one_hot.expand(-1, -1, H, W)  # [B, num_classes, H, W]

    return one_hot


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def create_synthetic_data(
    model, filtered_target_class_num_samples, device, batch_size, mode
) -> NDArrays:
    model.eval()
    model.to(device)

    # Create synthetic data in memory (or in batches if memory is constrained)
    synthetic_data = []
    synthetic_labels = []

    if mode == "HFedCVAE":
        latent_dim = metadata["HFedCVAE"]["vae_parameters"]["latent_dim"]
    elif mode == "HFedCGAN":
        latent_dim = metadata["HFedCGAN"]["generator_parameters"]["latent_dim"]
    elif mode == "HFedCVAEGAN":
        latent_dim = metadata["HFedCVAEGAN"]["vae_parameters"]["latent_dim"]

    for class_label, num_samples in filtered_target_class_num_samples.items():
        # Generate in batches to be memory efficient
        for start_idx in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - start_idx)

            images = []

            # Sample z
            z = torch.randn(current_batch_size, latent_dim, device=device)

            labels = torch.full(
                (current_batch_size,),
                class_label,
                dtype=torch.long,
                device=device,
            )

            if mode == "HFedCVAE" or mode == "HFedCVAEGAN":
                with torch.no_grad():
                    # One-hot labels (non-spatial)
                    one_hot = one_hot_labels(labels, model.num_classes)

                    # Concatenate z + label
                    z_with_label = torch.cat([z, one_hot], dim=1)

                    # Expand to feature map
                    expanded = model.fc_expand(z_with_label)
                    expanded = expanded.view(current_batch_size, *model.enc_shape)

                    # Spatial one-hot labels for decoder
                    decoder_one_hot_spatial = one_hot_labels(
                        labels,
                        model.num_classes,
                        (model.enc_shape[1], model.enc_shape[2]),
                    )

                    # Conditional decoder input
                    z_cond = torch.cat([expanded, decoder_one_hot_spatial], dim=1)

                    images = model.decoder(z_cond)
            elif mode == "HFedCGAN":
                with torch.no_grad():
                    images = model(z)

            synthetic_data.append(images.cpu())
            synthetic_labels.append(labels.cpu())

    # Concatenate all batches
    synthetic_data = torch.cat(synthetic_data, dim=0)
    synthetic_labels = torch.cat(synthetic_labels, dim=0)

    return [synthetic_data.numpy(), synthetic_labels.numpy()]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_total_data_generation_rounds(num_class_learn_per_round):
    return math.ceil(metadata["num_classes"] / num_class_learn_per_round)
