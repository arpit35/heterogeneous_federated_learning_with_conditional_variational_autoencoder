import torch.nn as nn

from src.scripts.helper import metadata


class GenericEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(metadata["encoder_input_dim"], metadata["encoder_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(metadata["encoder_hidden_dim"], metadata["encoder_hidden_dim"]),
            nn.ReLU(),
        )
        self.mu = nn.Linear(
            metadata["encoder_hidden_dim"], metadata["generic_encoder_latent_dim"]
        )
        self.logvar = nn.Linear(
            metadata["encoder_hidden_dim"], metadata["generic_encoder_latent_dim"]
        )

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)
