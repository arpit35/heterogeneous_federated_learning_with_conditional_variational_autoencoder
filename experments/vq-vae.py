import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading and Preprocessing
    def load_cifar10():
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Range [-0.5, 0.5]
            ]
        )

        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        # Split train into train and validation (45k train, 5k val)
        train_size = 45000
        val_size = 5000
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        return train_dataset, val_dataset, test_dataset

    train_dataset, val_dataset, test_dataset = load_cifar10()

    # Data variance for normalization (approximated from CIFAR-10)
    data_variance = (
        0.0627  # Approximate variance of CIFAR-10 images normalized to [0,1]
    )

    # Create data loaders with num_workers=0 to avoid multiprocessing issues
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Model Definitions
    class ResidualStack(nn.Module):
        def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
            super(ResidualStack, self).__init__()
            self.layers = nn.ModuleList()
            for i in range(num_residual_layers):
                self.layers.append(
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(
                            num_hiddens,
                            num_residual_hiddens,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        nn.ReLU(),
                        nn.Conv2d(
                            num_residual_hiddens, num_hiddens, kernel_size=1, stride=1
                        ),
                    )
                )

        def forward(self, x):
            for layer in self.layers:
                x = x + layer(x)
            return F.relu(x)

    class Encoder(nn.Module):
        def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
            super(Encoder, self).__init__()
            self.conv1 = nn.Conv2d(
                3, num_hiddens // 2, kernel_size=4, stride=2, padding=1
            )
            self.conv2 = nn.Conv2d(
                num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1
            )
            self.conv3 = nn.Conv2d(
                num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1
            )
            self.residual_stack = ResidualStack(
                num_hiddens, num_residual_layers, num_residual_hiddens
            )

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)
            x = self.residual_stack(x)
            return x

    class Decoder(nn.Module):
        def __init__(
            self, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim
        ):
            super(Decoder, self).__init__()
            # First conv to map from embedding_dim back to num_hiddens
            self.conv1 = nn.Conv2d(
                embedding_dim, num_hiddens, kernel_size=3, stride=1, padding=1
            )
            self.residual_stack = ResidualStack(
                num_hiddens, num_residual_layers, num_residual_hiddens
            )
            self.conv_trans1 = nn.ConvTranspose2d(
                num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1
            )
            self.conv_trans2 = nn.ConvTranspose2d(
                num_hiddens // 2, 3, kernel_size=4, stride=2, padding=1
            )

        def forward(self, x):
            x = self.conv1(x)  # Map from embedding_dim to num_hiddens
            x = self.residual_stack(x)
            x = F.relu(self.conv_trans1(x))
            x = self.conv_trans2(x)
            return x

    class VectorQuantizer(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, commitment_cost):
            super(VectorQuantizer, self).__init__()
            self.embedding_dim = embedding_dim
            self.num_embeddings = num_embeddings
            self.commitment_cost = commitment_cost

            self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
            self.embedding.weight.data.uniform_(
                -1 / self.num_embeddings, 1 / self.num_embeddings
            )

        def forward(self, inputs):
            # Convert inputs from BCHW -> BHWC
            inputs = inputs.permute(0, 2, 3, 1).contiguous()
            input_shape = inputs.shape

            # Flatten input
            flat_input = inputs.view(-1, self.embedding_dim)

            # Calculate distances
            distances = (
                torch.sum(flat_input**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2 * torch.matmul(flat_input, self.embedding.weight.t())
            )

            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(
                encoding_indices.shape[0], self.num_embeddings, device=inputs.device
            )
            encodings.scatter_(1, encoding_indices, 1)

            # Quantize and unflatten
            quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

            # Loss
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss

            quantized = inputs + (quantized - inputs).detach()
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            # Convert quantized from BHWC -> BCHW
            quantized = quantized.permute(0, 3, 1, 2).contiguous()

            return loss, quantized, perplexity, encodings

    class VQVAE(nn.Module):
        def __init__(
            self,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            num_embeddings,
            embedding_dim,
            commitment_cost,
        ):
            super(VQVAE, self).__init__()
            self.encoder = Encoder(
                num_hiddens, num_residual_layers, num_residual_hiddens
            )
            self.pre_vq_conv = nn.Conv2d(
                num_hiddens, embedding_dim, kernel_size=1, stride=1
            )
            self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            self.decoder = Decoder(
                num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim
            )

        def forward(self, x):
            z = self.encoder(x)
            z = self.pre_vq_conv(z)
            loss, quantized, perplexity, _ = self.vq(z)
            x_recon = self.decoder(quantized)
            return loss, x_recon, perplexity

    # Hyperparameters
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25
    learning_rate = 3e-4
    num_training_updates = (
        50000  # Reduced for testing, change back to 50000 for full training
    )

    # Model, optimizer
    model = VQVAE(
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        num_embeddings,
        embedding_dim,
        commitment_cost,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Print model architecture
    print("Model architecture:")
    print(model)

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Training
    train_res_recon_error = []
    train_res_perplexity = []

    # Create an infinite iterator for the training data
    train_iter = cycle(train_loader)

    model.train()
    print("Starting training...")
    for i in range(num_training_updates):
        data, _ = next(train_iter)
        data = data.to(device)

        optimizer.zero_grad()
        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()
        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i + 1) % 100 == 0:
            print(f"{i + 1} iterations")
            print(f"recon_error: {np.mean(train_res_recon_error[-100:]):.3f}")
            print(f"perplexity: {np.mean(train_res_perplexity[-100:]):.3f}")
            print()

    # Plotting
    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(1, 2, 1)
    ax.plot(train_res_recon_error)
    ax.set_yscale("log")
    ax.set_title("NMSE")

    ax = f.add_subplot(1, 2, 2)
    ax.plot(train_res_perplexity)
    ax.set_title("Average codebook usage (perplexity)")
    plt.close()

    # Reconstructions - FIXED VERSION
    def convert_batch_to_image_grid(image_batch):
        """
        Convert a batch of images to a grid for visualization.
        image_batch: numpy array of shape (batch_size, channels, height, width)
        """
        # Denormalize: from [-0.5, 0.5] to [0, 1]
        image_batch = image_batch * 0.5 + 0.5

        # Convert to numpy and ensure correct data type
        if isinstance(image_batch, torch.Tensor):
            image_batch = image_batch.detach().cpu().numpy()

        # Make sure values are in [0, 1] range
        image_batch = np.clip(image_batch, 0, 1)

        # Reorder dimensions from (batch, channels, height, width) to (batch, height, width, channels)
        image_batch = image_batch.transpose(0, 2, 3, 1)

        # Take first 32 images and reshape to grid: 4 rows, 8 columns
        batch_size = min(32, image_batch.shape[0])
        grid = image_batch[:batch_size]

        # Reshape to grid
        n_rows = 4
        n_cols = 8
        grid = grid.reshape(n_rows, n_cols, 32, 32, 3)

        # Combine into single image
        grid = grid.transpose(0, 2, 1, 3, 4)  # (4, 32, 8, 32, 3)
        grid = grid.reshape(n_rows * 32, n_cols * 32, 3)

        return grid

    def get_reconstructions(loader, model, num_batches=1):
        model.eval()
        originals = []
        reconstructions = []

        with torch.no_grad():
            for i, (data, _) in enumerate(loader):
                if i >= num_batches:
                    break
                data = data.to(device)
                _, recon, _ = model(data)

                originals.append(data.cpu())
                reconstructions.append(recon.cpu())

        # Convert to numpy arrays
        originals = torch.cat(originals, dim=0).numpy()
        reconstructions = torch.cat(reconstructions, dim=0).numpy()

        return originals, reconstructions

    print("Generating reconstructions...")
    train_originals, train_reconstructions = get_reconstructions(train_loader, model)
    val_originals, val_reconstructions = get_reconstructions(val_loader, model)

    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Training originals
    axes[0, 0].imshow(convert_batch_to_image_grid(train_originals))
    axes[0, 0].set_title("Training Data Originals")
    axes[0, 0].axis("off")

    # Training reconstructions
    axes[0, 1].imshow(convert_batch_to_image_grid(train_reconstructions))
    axes[0, 1].set_title("Training Data Reconstructions")
    axes[0, 1].axis("off")

    # Validation originals
    axes[1, 0].imshow(convert_batch_to_image_grid(val_originals))
    axes[1, 0].set_title("Validation Data Originals")
    axes[1, 0].axis("off")

    # Validation reconstructions
    axes[1, 1].imshow(convert_batch_to_image_grid(val_reconstructions))
    axes[1, 1].set_title("Validation Data Reconstructions")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
