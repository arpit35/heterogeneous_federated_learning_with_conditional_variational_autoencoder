import torch
import torch.nn as nn

from src.ml_models.residual import ResidualStack
from src.scripts.helper import metadata


class Discriminator(nn.Module):
    def __init__(
        self,
        in_dim=metadata["num_channels"],
        h_dim=metadata["h_dim"] // 4,
        n_res_layers=max(metadata["n_res_layers"] // 4, 1),
        res_h_dim=metadata["res_h_dim"] // 4,
    ):
        super(Discriminator, self).__init__()
        kernel = 4
        stride = 2

        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1
            ),
            nn.ReLU(),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                metadata["num_channels"],
                metadata["image_width"],
                metadata["image_height"],
            )
            conv_stack_out = self.conv_stack(dummy)
            _, C_enc, H_enc, W_enc = conv_stack_out.shape

        self.fc1 = nn.Linear(C_enc * H_enc * W_enc, 1)

    def forward(self, x):
        x = self.conv_stack(x)
        return self.fc1(x.view(x.size(0), -1))
