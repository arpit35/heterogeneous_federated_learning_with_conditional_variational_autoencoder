import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== Configuration ====================
class Config:
    # Dataset
    image_size = 32
    num_classes = 10

    # Diffusion
    timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02

    # Model
    base_channels = 64
    channel_mults = [1, 2, 4, 8]
    num_res_blocks = 2
    dropout = 0.1
    attention_resolutions = [16]

    # Training
    batch_size = 64
    learning_rate = 1e-4
    epochs = 50

    # Sampling
    sampling_timesteps = 250  # DDIM sampling steps


config = Config()


# ==================== Diffusion Scheduler ====================
class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device
        )
        self.posterior_variance = self.posterior_variance.to(device)

    def add_noise(self, x_start, t, noise=None):
        """Forward diffusion: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1
        )

        return (
            sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise,
            noise,
        )

    def sample_prev_timestep(self, model_output, x_t, t, t_index):
        """Reverse diffusion step: p(x_{t-1} | x_t)"""
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1
        )
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).reshape(-1, 1, 1, 1)

        # Equation 11 in DDPM paper
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise


# ==================== U-Net Components ====================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        self.register_buffer("embeddings", embeddings)

    def forward(self, time):
        time = time.unsqueeze(1)
        embeddings = time * self.embeddings.unsqueeze(0)
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, dim)

    def forward(self, labels):
        return self.embedding(labels)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, label_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)
        self.label_proj = nn.Linear(label_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb, label_emb):
        residual = self.residual_conv(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time_emb = self.time_proj(F.silu(time_emb))
        label_emb = self.label_proj(F.silu(label_emb))
        emb = time_emb + label_emb

        h = h + emb[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        residual = x
        b, c, h, w = x.shape

        x = self.norm(x)

        q = self.q(x).reshape(b, c, -1).transpose(1, 2)
        k = self.k(x).reshape(b, c, -1)
        v = self.v(x).reshape(b, c, -1).transpose(1, 2)

        attn = torch.bmm(q, k) * (c**-0.5)
        attn = F.softmax(attn, dim=-1)

        h = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        h = self.proj_out(h)

        return h + residual


class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        time_dim = config.base_channels * 4
        label_dim = config.base_channels * 4

        self.time_embed = TimeEmbedding(time_dim)
        self.label_embed = LabelEmbedding(config.num_classes, label_dim)

        # Initial convolution
        self.init_conv = nn.Conv2d(1, config.base_channels, 3, padding=1)

        # Down blocks
        self.down_blocks = nn.ModuleList()
        channels = config.base_channels
        input_block_chans = [channels]

        for i, mult in enumerate(config.channel_mults):
            out_channels = config.base_channels * mult
            for _ in range(config.num_res_blocks):
                self.down_blocks.append(
                    ResBlock(
                        channels, out_channels, time_dim, label_dim, config.dropout
                    )
                )
                channels = out_channels
                input_block_chans.append(channels)

                if i < len(config.attention_resolutions):
                    size = config.image_size // (2**i)
                    if size in config.attention_resolutions:
                        self.down_blocks.append(AttentionBlock(channels))

            if i != len(config.channel_mults) - 1:
                self.down_blocks.append(
                    nn.Conv2d(channels, channels, 3, stride=2, padding=1)
                )
                input_block_chans.append(channels)

        # Middle blocks
        self.mid_block1 = ResBlock(
            channels, channels, time_dim, label_dim, config.dropout
        )
        self.mid_attn = AttentionBlock(channels)
        self.mid_block2 = ResBlock(
            channels, channels, time_dim, label_dim, config.dropout
        )

        # Up blocks
        self.up_blocks = nn.ModuleList()

        for i, mult in reversed(list(enumerate(config.channel_mults))):
            out_channels = config.base_channels * mult
            for j in range(config.num_res_blocks + 1):
                skip_channels = input_block_chans.pop()
                self.up_blocks.append(
                    ResBlock(
                        channels + skip_channels,
                        out_channels,
                        time_dim,
                        label_dim,
                        config.dropout,
                    )
                )
                channels = out_channels

                if i < len(config.attention_resolutions):
                    size = config.image_size // (2**i)
                    if size in config.attention_resolutions:
                        self.up_blocks.append(AttentionBlock(channels))

            if i != 0:
                self.up_blocks.append(
                    nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
                )

        # Final
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x, timesteps, labels):
        # Embeddings
        t_emb = self.time_embed(timesteps)
        l_emb = self.label_embed(labels)

        # Initial convolution
        h = self.init_conv(x)
        hs = [h]

        # Downsample
        for module in self.down_blocks:
            if isinstance(module, ResBlock):
                h = module(h, t_emb, l_emb)
            else:
                h = module(h)
            hs.append(h)

        # Middle
        h = self.mid_block1(h, t_emb, l_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb, l_emb)

        # Upsample
        for module in self.up_blocks:
            if isinstance(module, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, t_emb, l_emb)
            elif isinstance(module, AttentionBlock):
                h = module(h)
            else:
                h = module(h)

        # Final
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


# ==================== Training Function ====================
def train(model, dataloader, scheduler, optimizer, epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Sample random timesteps
        t = torch.randint(0, config.timesteps, (images.shape[0],), device=device).long()

        # Sample noise
        noise = torch.randn_like(images)

        # Add noise
        noisy_images, noise_target = scheduler.add_noise(images, t, noise)

        # Predict noise
        noise_pred = model(noisy_images, t, labels)

        # Compute loss
        loss = F.mse_loss(noise_pred, noise_target)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


# ==================== Sampling Function ====================
@torch.no_grad()
def sample(model, scheduler, labels, num_samples=1):
    model.eval()

    # Start from pure noise
    x = torch.randn(
        (num_samples, 1, config.image_size, config.image_size), device=device
    )
    labels = labels.to(device)

    # DDIM sampling (simplified)
    indices = list(
        range(0, config.timesteps, config.timesteps // config.sampling_timesteps)
    )

    for i in tqdm(reversed(range(len(indices))), desc="Sampling"):
        t = torch.full((num_samples,), indices[i], device=device, dtype=torch.long)

        # Predict noise
        noise_pred = model(x, t, labels)

        # Remove noise
        alpha = scheduler.alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, 1, 1, 1
        )

        x = (x - sqrt_one_minus_alpha * noise_pred) / torch.sqrt(alpha)

        if i > 0:
            sigma = 0.0  # DDIM
            noise = torch.randn_like(x)
            x = x + sigma * noise

    return x.clamp(-1, 1)


# ==================== Main Training Loop ====================
def main():
    # Prepare dataset
    transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # MNIST mean and std
        ]
    )

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2
    )

    # Initialize model, scheduler, and optimizer
    model = UNet(config).to(device)
    scheduler = DiffusionScheduler(config.timesteps, config.beta_start, config.beta_end)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop
    print("Starting training...")
    for epoch in range(1, config.epochs + 1):
        loss = train(model, train_loader, scheduler, optimizer, epoch)
        print(f"Epoch {epoch}/{config.epochs}, Loss: {loss:.4f}")

        # Generate samples every 10 epochs
        if epoch % 10 == 0 or epoch == config.epochs:
            # Generate one sample per class
            with torch.no_grad():
                labels = torch.arange(10, device=device).repeat(
                    2
                )  # 2 samples per class
                samples = sample(model, scheduler, labels, num_samples=20)

                # Plot generated samples
                fig, axes = plt.subplots(2, 10, figsize=(15, 3))
                samples = samples.cpu().numpy()

                for i in range(2):
                    for j in range(10):
                        idx = i * 10 + j
                        ax = axes[i, j]
                        img = samples[idx, 0]
                        ax.imshow(img, cmap="gray")
                        ax.axis("off")
                        ax.set_title(f"Label: {labels[idx].item()}")

                plt.suptitle(f"Epoch {epoch} - Generated MNIST Digits")
                plt.tight_layout()
                plt.savefig(f"samples_epoch_{epoch}.png", dpi=100, bbox_inches="tight")
                plt.close()

                # Save model checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    f"stable_diffusion_mnist_epoch_{epoch}.pth",
                )

    print("Training completed!")

    # Final generation example
    print("\nGenerating final samples...")

    # Generate specific digits
    specific_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], device=device)
    samples = sample(model, scheduler, specific_labels, num_samples=10)

    # Display results
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    samples = samples.cpu().numpy()

    for i in range(2):
        for j in range(5):
            idx = i * 5 + j
            ax = axes[i, j]
            img = samples[idx, 0]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            ax.set_title(f"Generated: {specific_labels[idx].item()}")

    plt.suptitle("Stable Diffusion Generated MNIST Digits")
    plt.tight_layout()
    plt.show()

    # Save final model
    torch.save(model.state_dict(), "stable_diffusion_mnist_final.pth")
    print("Model saved as 'stable_diffusion_mnist_final.pth'")


# ==================== Quick Demo ====================
def quick_demo():
    """Quick demonstration with a smaller model for faster results"""
    print("Running quick demo with smaller model...")

    # Smaller config for demo
    demo_config = Config()
    demo_config.image_size = 28
    demo_config.timesteps = 500
    demo_config.base_channels = 32
    demo_config.channel_mults = [1, 2, 4]
    demo_config.batch_size = 32
    demo_config.epochs = 10

    # Load MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=demo_config.batch_size, shuffle=True
    )

    # Initialize model
    model = UNet(demo_config).to(device)
    scheduler = DiffusionScheduler(
        demo_config.timesteps, demo_config.beta_start, demo_config.beta_end
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train for a few epochs
    print("Training demo model (this will take a few minutes)...")
    for epoch in range(1, 4):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            t = torch.randint(
                0, demo_config.timesteps, (images.shape[0],), device=device
            ).long()
            noise = torch.randn_like(images)
            noisy_images, noise_target = scheduler.add_noise(images, t, noise)

            noise_pred = model(noisy_images, t, labels)
            loss = F.mse_loss(noise_pred, noise_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Demo Epoch {epoch}, Loss: {loss.item():.4f}")

    # Generate samples
    print("\nGenerating demo samples...")
    model.eval()

    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], device=device)

    with torch.no_grad():
        x = torch.randn(
            (10, 1, demo_config.image_size, demo_config.image_size), device=device
        )

        for t in tqdm(reversed(range(demo_config.timesteps)), desc="Sampling"):
            t_batch = torch.full((10,), t, device=device, dtype=torch.long)
            noise_pred = model(x, t_batch, labels)

            alpha = scheduler.alphas_cumprod[t].reshape(1, 1, 1, 1)
            sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t].reshape(
                1, 1, 1, 1
            )

            x = (x - sqrt_one_minus_alpha * noise_pred) / torch.sqrt(alpha)

            if t > 0:
                noise = torch.randn_like(x)
                x = x + 0.1 * noise

        samples = x.clamp(-1, 1).cpu().numpy()

    # Display
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(2):
        for j in range(5):
            idx = i * 5 + j
            ax = axes[i, j]
            img = samples[idx, 0]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            ax.set_title(f"Label: {labels[idx].item()}")

    plt.suptitle("Demo: Stable Diffusion Generated MNIST")
    plt.tight_layout()
    plt.show()

    print("Demo completed!")


# ==================== Run ====================
if __name__ == "__main__":
    # For full training, run:
    # main()

    # For quick demo (recommended for first run):
    quick_demo()
