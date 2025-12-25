import torch
import torch.nn as nn

from src.scripts.helper import metadata


class Generator(nn.Module):
    """
    DCGAN Generator
    z -> image
    """

    def __init__(
        self,
        block_repeat,
        n_block_layers,
        h_dim,
        latent_dim,
        out_dim=metadata["num_channels"],
    ):
        super(Generator, self).__init__()

        kernel_size = 4

        def generator_block(in_filters, out_filters, kernel_size=4, padding=1):
            block = [
                nn.ConvTranspose2d(
                    in_filters,
                    out_filters,
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                nn.BatchNorm2d(out_filters),
                nn.ReLU(True),
            ]
            return block

        self.init_img_dim = metadata["image_width"] - (
            n_block_layers * (kernel_size - 1)
        )
        self.fc1 = nn.Linear(latent_dim, self.init_img_dim**2)

        block_h_dim = h_dim
        generator_block_list = generator_block(1, block_h_dim, kernel_size=3, padding=1)

        for _ in range(n_block_layers):

            if block_repeat > 1:
                for _ in range(block_repeat - 1):
                    generator_block_list.extend(
                        generator_block(
                            block_h_dim, block_h_dim, kernel_size=3, padding=1
                        )
                    )

            generator_block_list.extend(
                generator_block(block_h_dim, block_h_dim // 2, kernel_size, 0)
            )

            block_h_dim = block_h_dim // 2

        self.conv_t_stack = nn.Sequential(
            *generator_block_list,
            nn.ConvTranspose2d(block_h_dim, out_dim, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.size(0), -1, self.init_img_dim, self.init_img_dim)
        return self.conv_t_stack(x)
