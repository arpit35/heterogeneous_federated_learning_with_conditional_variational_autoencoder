"""
=======================
GRADIENT FLOW EXPLANATION
=======================

We train the Discriminator (D) and Generator (G) in two separate steps.
The key idea is controlling how the computation graph connects G and D.

-----------------------
1. DISCRIMINATOR STEP
-----------------------
optimizer_D.zero_grad()

# a) Real images:
real_imgs -> D -> real_pred
Loss = BCE(real_pred, label=1)
Gradients: Loss -> D (update D's params)

# b) Fake images:
z -> G -> gen_imgs.detach() -> D -> fake_pred
Loss = BCE(fake_pred, label=0)
Here `.detach()` is CRUCIAL:
    - It breaks the graph between G and D.
    - This means no gradient flows back into G during D's update.
    - Only D's parameters are updated here.

# Combine real and fake losses:
d_loss = (loss_real + loss_fake) / 2
d_loss.backward()
optimizer_D.step()
Result: D gets better at distinguishing real from fake.

-------------------
2. GENERATOR STEP
-------------------
optimizer_G.zero_grad()

z -> G -> gen_imgs -> D -> validity
Loss = BCE(validity, label=1)  # G wants D to think fakes are real

# IMPORTANT:
No `.detach()` here, so the computation graph from D's input
(fakes) back to G's parameters is intact.
Gradients flow:
    BCE loss -> D (frozen, but passes gradients)
               -> gen_imgs
               -> G (updates its params)

# Even though gradients pass through D, its parameters
are NOT updated in this step because we only call optimizer_G.step().

g_loss.backward()
optimizer_G.step()
Result: G learns to produce more realistic fakes based on D's feedback.

-------------------
GRADIENT FLOW SUMMARY
-------------------
- D step: Gradients flow ONLY inside D (G is cut off with detach()).
- G step: Gradients flow through D INTO G, but only G's params update.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 128
latent_dim = 100
learning_rate = 0.0002
epochs = 50
img_size = 32
img_channels = 1

# Create output directory
os.makedirs("dcgan_output", exist_ok=True)

# Enhanced preprocessing with resizing
transform = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


# DCGAN Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = 28 // 4

        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# DCGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = img_size // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


if __name__ == "__main__":
    # Load dataset
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
    )

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Loss function and optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(
        generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )

    # Fixed noise for consistent sample evaluation
    fixed_noise = torch.randn(64, latent_dim, device=device)

    # Training statistics
    G_losses = []
    D_losses = []

    # Training loop
    start_time = time.time()
    print("Starting Training...")

    for epoch in range(epochs):
        epoch_start = time.time()

        for i, (imgs, _) in enumerate(train_loader):
            # Configure input
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)

            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real images
            real_pred = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, valid)

            # Fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z)
            fake_pred = discriminator(gen_imgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate images and calculate loss
            validity = discriminator(gen_imgs)
            g_loss = adversarial_loss(validity, valid)
            g_loss.backward()
            optimizer_G.step()

            # Save losses
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

            # Print progress
            if i % 200 == 0:
                print(
                    f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )

        # Generate sample images at end of epoch
        with torch.no_grad():
            gen_samples = generator(fixed_noise)
            save_image(
                gen_samples.data,
                f"dcgan_output/epoch_{epoch}.png",
                nrow=8,
                normalize=True,
            )

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")

        # Save model checkpoints every 10 epochs
        if epoch % 10 == 0:
            torch.save(
                generator.state_dict(), f"dcgan_output/generator_epoch_{epoch}.pth"
            )
            torch.save(
                discriminator.state_dict(),
                f"dcgan_output/discriminator_epoch_{epoch}.pth",
            )

    total_time = time.time() - start_time
    print(f"Training completed in {total_time//60:.0f}m {total_time%60:.0f}s")

    # Generate final samples
    def generate_final_samples():
        generator.eval()
        with torch.no_grad():
            z = torch.randn(64, latent_dim, device=device)
            samples = generator(z).cpu()
            samples = samples * 0.5 + 0.5

        fig, axes = plt.subplots(8, 8, figsize=(12, 12))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(samples[i].permute(1, 2, 0).squeeze(), cmap="gray")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig("dcgan_output/final_samples.png")
        plt.show()

    generate_final_samples()

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("dcgan_output/loss_curve.png")
    plt.show()
