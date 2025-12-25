from collections import OrderedDict

import torch
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
    vae, label, device, samples_per_class, batch_size
) -> NDArrays:
    vae.eval()
    vae.to(device)

    # Create synthetic data in memory (or in batches if memory is constrained)
    synthetic_data = []
    synthetic_labels = []

    # Generate in batches to be memory efficient
    for start_idx in range(0, samples_per_class, batch_size):
        current_batch_size = min(batch_size, samples_per_class - start_idx)

        # Sample z
        z = torch.randn(current_batch_size, metadata["latent_dim"], device=device)

        # Decode
        with torch.no_grad():
            expanded = vae.fc_expand(z)
            expanded = expanded.view(current_batch_size, *vae.enc_shape)
            images = vae.decoder(expanded)

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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
