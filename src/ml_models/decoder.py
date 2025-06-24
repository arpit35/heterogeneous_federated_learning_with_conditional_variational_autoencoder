import torch
import torch.nn as nn

from src.scripts.helper import metadata


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                metadata["generic_encoder_latent_dim"]
                + metadata["personalized_encoder_latent_dim"],
                metadata["decoder_hidden_dim"],
            ),
            nn.ReLU(),
            nn.Linear(metadata["decoder_hidden_dim"], metadata["decoder_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(
                metadata["decoder_hidden_dim"],
                metadata["image_width"]
                * metadata["image_height"]
                * metadata["num_channels"],
            ),
            nn.Tanh(),
        )

    def forward(self, generic_encoder_latent_space, personalized_encoder_latent_space):
        return self.net(
            torch.cat(
                [generic_encoder_latent_space, personalized_encoder_latent_space], dim=1
            )
        )
