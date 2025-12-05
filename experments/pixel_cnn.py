import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 4

# Metadata for MNIST
metadata = {
    "n_embeddings": 256,  # 256 pixel values (0-255)
    "embedding_dim": 64,
    "n_pixel_cnn_layers": 15,
    "num_classes": 10,  # MNIST has 10 digits
}


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            pass


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()
        assert kernel % 2 == 1, "Kernel size must be odd"
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(n_classes, 2 * dim)

        kernel_shp = (kernel // 2 + 1, kernel)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(dim, dim * 2, kernel_shp, 1, padding_shp)

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(dim, dim * 2, kernel_shp, 1, padding_shp)

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()
        self.horiz_stack.weight.data[:, :, :, -1].zero_()

    def forward(self, x_v, x_h, h):
        if self.mask_type == "A":
            self.make_causal()

        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, : x_v.size(-1), :]
        out_v = self.gate(h_vert + h[:, :, None, None])

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, : x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(
        self,
        input_dim=256,
        dim=64,
        n_layers=15,
        n_classes=10,
    ):
        super().__init__()
        self.dim = dim

        self.embedding = nn.Embedding(input_dim, dim)

        self.layers = nn.ModuleList()

        for i in range(n_layers):
            mask_type = "A" if i == 0 else "B"
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
            )

        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1), nn.ReLU(True), nn.Conv2d(512, input_dim, 1)
        )

        self.apply(weights_init)

    def forward(self, x, label):
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)
        x = x.permute(0, 3, 1, 2)

        x_v, x_h = (x, x)
        for layer in self.layers:
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    def generate(self, label, shape=(28, 28), batch_size=64):
        self.eval()
        with torch.no_grad():
            param = next(self.parameters())
            x = torch.zeros(
                (batch_size, *shape), dtype=torch.int64, device=param.device
            )

            for i in range(shape[0]):
                for j in range(shape[1]):
                    logits = self.forward(x, label)
                    probs = F.softmax(logits[:, :, i, j], -1)
                    x.data[:, i, j].copy_(probs.multinomial(1).squeeze().data)

            self.train()
            return x


def _convert_to_int(x):
    """Convert image tensor to integers 0-255"""
    return (x * 255).long()


def download_and_load_mnist():
    """Download and load MNIST dataset"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(_convert_to_int),  # Convert to integers 0-255
        ]
    )

    # Download training dataset
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Download test dataset
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Create data loaders (num_workers=0 for Windows compatibility)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    return train_loader, test_loader


def train_pixelcnn(model, train_loader, test_loader, epochs=EPOCHS):
    """Train the PixelCNN model"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_losses = []
    test_losses = []

    print("Starting training...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            data = data.squeeze(1)  # Remove channel dimension (B, 1, H, W) -> (B, H, W)

            optimizer.zero_grad()
            outputs = model(data, targets)

            # Reshape for cross entropy loss
            outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, 256)
            targets_flat = data.view(-1)

            loss = F.cross_entropy(outputs, targets_flat)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Testing phase
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                data = data.squeeze(1)

                outputs = model(data, targets)
                outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, 256)
                targets_flat = data.view(-1)

                loss = F.cross_entropy(outputs, targets_flat)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        scheduler.step(avg_test_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}"
        )

    return train_losses, test_losses


def generate_synthetic_data(model, labels, n_samples_per_label=10):
    """Generate synthetic MNIST digits for given labels"""
    model.eval()
    generated_images = []

    for label in labels:
        # Create labels tensor
        label_tensor = torch.full((n_samples_per_label,), label, dtype=torch.long).to(
            device
        )

        # Generate images
        with torch.no_grad():
            synthetic = model.generate(
                label_tensor, shape=(28, 28), batch_size=n_samples_per_label
            )

        # Convert to numpy for visualization
        synthetic_np = synthetic.cpu().numpy() / 255.0  # Normalize to [0, 1]
        generated_images.append((label, synthetic_np))

    return generated_images


def plot_results(generated_images):
    """Plot generated images"""
    n_labels = len(generated_images)
    n_samples = len(generated_images[0][1])

    fig, axes = plt.subplots(n_labels, n_samples, figsize=(15, 5))
    if n_labels == 1:
        axes = axes.reshape(1, -1)

    for i, (label, images) in enumerate(generated_images):
        for j in range(n_samples):
            ax = axes[i, j] if n_labels > 1 else axes[j]
            ax.imshow(images[j], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if j == 0:
                ax.set_title(f"Label: {label}")

    plt.tight_layout()
    plt.savefig("generated_mnist.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(train_losses, test_losses):
    """Plot training and test loss history"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    print(f"Using device: {device}")

    # 1. Download and load MNIST
    print("Downloading MNIST dataset...")
    train_loader, test_loader = download_and_load_mnist()

    # 2. Initialize model
    print("Initializing PixelCNN model...")
    model = GatedPixelCNN(
        input_dim=metadata["n_embeddings"],
        dim=metadata["embedding_dim"],
        n_layers=metadata["n_pixel_cnn_layers"],
        n_classes=metadata["num_classes"],
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")

    # 3. Train model
    print("\nTraining PixelCNN...")
    train_losses, test_losses = train_pixelcnn(model, train_loader, test_loader)

    # Plot training history
    plot_training_history(train_losses, test_losses)

    # 4. Generate synthetic data
    print("\nGenerating synthetic MNIST digits...")

    # Generate for digits 0-9
    labels_to_generate = list(range(10))
    generated_images = generate_synthetic_data(
        model, labels_to_generate, n_samples_per_label=5
    )

    # 5. Visualize results
    print("Plotting generated images...")
    plot_results(generated_images)

    # Save model
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": metadata},
        "pixelcnn_mnist.pth",
    )
    print("Model saved to pixelcnn_mnist.pth")

    # Print some statistics
    print("\n=== Generation Statistics ===")
    for label, images in generated_images:
        print(f"Label {label}: Generated {len(images)} images")
        print(f"  Pixel value range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Mean pixel value: {images.mean():.3f}")
        print()


if __name__ == "__main__":
    main()
