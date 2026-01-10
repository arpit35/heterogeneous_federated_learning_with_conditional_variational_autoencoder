import torch
import torch.nn as nn

from src.ml_models.utils import one_hot_labels
from src.scripts.helper import metadata


class Generator(nn.Module):
    """
    DCGAN Generator
    z -> image
    """

    def __init__(
        self,
        n_block_layers,
        h_dim,
        latent_dim,
        init_img_dim,
        num_classes=metadata["num_classes"],
        out_dim=metadata["num_channels"],
    ):
        super(Generator, self).__init__()

        def generator_block(in_filters, out_filters):
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.init_img_dim = init_img_dim
        self.num_classes = num_classes

        block_h_dim = h_dim
        self.fc1 = nn.Linear(
            latent_dim + num_classes, block_h_dim * self.init_img_dim**2
        )

        blocks = []

        for _ in range(n_block_layers):
            blocks.append(generator_block(block_h_dim, block_h_dim // 2))

            block_h_dim = block_h_dim // 2

        allignment_correction = metadata["image_width"] - (
            self.init_img_dim * (2**n_block_layers)
        )

        self.conv_t_stack = nn.Sequential(
            *blocks,
            nn.ConvTranspose2d(
                block_h_dim,
                out_dim,
                kernel_size=3 + allignment_correction,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        one_hot = one_hot_labels(labels, self.num_classes)
        x = torch.cat([x, one_hot], dim=1)

        x = self.fc1(x)
        x = x.view(x.size(0), -1, self.init_img_dim, self.init_img_dim)
        return self.conv_t_stack(x)
