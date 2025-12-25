import torch
import torch.nn as nn

from src.scripts.helper import metadata


class Discriminator(nn.Module):
    def __init__(
        self,
        block_repeat,
        n_block_layers,
        h_dim,
        in_dim=metadata["num_channels"],
    ):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, bn=True):
            block = [
                nn.Conv2d(
                    in_filters, out_filters, kernel_size=3, stride=stride, padding=1
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.3),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))

            return block

        block_h_dim = h_dim
        discriminator_block_list = discriminator_block(
            in_dim, block_h_dim, stride=1, bn=False
        )

        for _ in range(n_block_layers):

            if block_repeat > 1:
                for _ in range(block_repeat - 1):
                    discriminator_block_list.extend(
                        discriminator_block(block_h_dim, block_h_dim, 1)
                    )

            discriminator_block_list.extend(
                discriminator_block(block_h_dim, block_h_dim // 2)
            )

            block_h_dim = block_h_dim // 2

        self.conv_stack = nn.Sequential(*discriminator_block_list)

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
