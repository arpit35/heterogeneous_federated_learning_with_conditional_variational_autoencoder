import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim = 784  # 28x28 flattened
hidden_dim = 400
latent_z = 25  # Universal knowledge representation
latent_c = 3  # Client-specific representation
batch_size = 128
epochs = 20
learning_rate = 0.001
alpha = 1.0  # Weight for z regularizer
beta = 1.0  # Weight for c regularizer
xi_k = 0.1  # Constraint threshold


# =================================================================
# DATASET PREPARATION WITH CIRCLES
# =================================================================
def add_circle(img_tensor):
    """Add a random black circle outline to the image tensor"""
    img = img_tensor.clone().squeeze(0)  # Remove channel dim
    h, w = img.shape

    # Random circle parameters
    x0 = torch.randint(5, 23, (1,)).item()
    y0 = torch.randint(5, 23, (1,)).item()
    r = torch.randint(3, 8, (1,)).item()

    # Create grid
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    # Calculate distance squared from center
    dist_sq = (x - x0) ** 2 + (y - y0) ** 2

    # Create outline mask
    outer_mask = dist_sq <= (r + 0.5) ** 2
    inner_mask = dist_sq <= (r - 0.5) ** 2
    circle_mask = outer_mask & ~inner_mask

    # Set circle outline pixels to 1.0 (white)
    img[circle_mask] = 1.0

    return img.unsqueeze(0)  # Add back channel dim


# MNIST Dataset with circles
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: add_circle(x)),  # Add circle
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten
    ]
)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# =================================================================
# FEDDVA MODEL IMPLEMENTATION (CENTRALIZED)
# =================================================================
class UniversalEncoder(nn.Module):
    """q(z|x) = N(z; μ(x), Σ(x))"""

    def __init__(self, input_dim, hidden_dim, latent_z):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_z)
        self.logvar = nn.Linear(hidden_dim, latent_z)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class ClientEncoder(nn.Module):
    """q(c|x,z) = N(c; μ̂(x,z), Σ̂(x,z))"""

    def __init__(self, input_dim, latent_z, hidden_dim, latent_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + latent_z, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_c)
        self.logvar = nn.Linear(hidden_dim, latent_c)

    def forward(self, x, z):
        h = self.net(torch.cat([x, z], dim=1))
        return self.mu(h), self.logvar(h)


class ClientDecoder(nn.Module):
    """p_k(x|z,c) - Client-specific decoder"""

    def __init__(self, latent_z, latent_c, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_z + latent_c, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))


class FedDVA(nn.Module):
    """FedDVA complete model (centralized version)"""

    def __init__(self, input_dim, hidden_dim, latent_z, latent_c):
        super().__init__()
        self.universal_encoder = UniversalEncoder(input_dim, hidden_dim, latent_z)
        self.client_encoder = ClientEncoder(input_dim, latent_z, hidden_dim, latent_c)
        self.client_decoder = ClientDecoder(latent_z, latent_c, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Universal representation
        mu_z, logvar_z = self.universal_encoder(x)
        z = self.reparameterize(mu_z, logvar_z)

        # Client-specific representation
        mu_c, logvar_c = self.client_encoder(x, z)
        c = self.reparameterize(mu_c, logvar_c)

        # Reconstruction
        recon_x = self.client_decoder(z, c)

        return recon_x, z, c, mu_z, logvar_z, mu_c, logvar_c


# =================================================================
# LOSS FUNCTION WITH KL CONSTRAINTS (Eq.4-7)
# =================================================================
def feddva_loss(
    recon_x,
    x,
    mu_z,
    logvar_z,
    mu_c,
    logvar_c,
    all_mu_c,
    all_logvar_c,
    alpha,
    beta,
    xi_k,
):
    # Reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # KL Regularizers (Eq.6-7)
    # R_z: KL(q(z|x) || p(z)) (p(z) = N(0,I))
    KLD_z = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

    # Compute p_k(c) as mixture distribution (Eq.5)
    mu_c_mix = torch.mean(all_mu_c, dim=0)
    logvar_c_mix = torch.mean(all_logvar_c, dim=0)

    # KL(q(c|x,z) || p(c)) (p(c) = N(0,I))
    KLD_c_prior = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())

    # KL(q(c|x,z) || p_k(c))
    var_c = logvar_c.exp()
    var_c_mix = logvar_c_mix.exp()
    KLD_c_client = 0.5 * torch.sum(
        logvar_c_mix - logvar_c + (var_c + (mu_c - mu_c_mix).pow(2)) / var_c_mix - 1
    )

    # Slack regularizer (Eq.5,7)
    constraint = xi_k + KLD_c_client
    R_c = torch.max(constraint, KLD_c_prior)

    # Final loss (Eq.4)
    loss = BCE + alpha * KLD_z + beta * R_c

    return loss, BCE, KLD_z, R_c


# Initialize model
model = FedDVA(input_dim, hidden_dim, latent_z, latent_c).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# =================================================================
# TRAINING LOOP
# =================================================================
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        recon_x, z, c, mu_z, logvar_z, mu_c, logvar_c = model(data)

        # Collect statistics for p_k(c) from current batch
        all_mu_c = mu_c.detach().clone()
        all_logvar_c = logvar_c.detach().clone()

        # Calculate loss
        loss, BCE, KLD_z, R_c = feddva_loss(
            recon_x,
            data,
            mu_z,
            logvar_z,
            mu_c,
            logvar_c,
            all_mu_c,
            all_logvar_c,
            alpha,
            beta,
            xi_k,
        )

        # Backpropagation
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item()/len(data):.6f}"
            )

    avg_loss = train_loss / len(train_loader.dataset)
    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")
    return avg_loss


# Test function
def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_x, z, c, mu_z, logvar_z, mu_c, logvar_c = model(data)

            # For test loss, use standard VAE loss since we don't have batch statistics
            # for p_k(c) across entire dataset
            BCE = F.binary_cross_entropy(recon_x, data, reduction="sum")
            KLD_z = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
            KLD_c = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())
            loss = BCE + KLD_z + KLD_c

            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print(f"====> Test set loss: {test_loss:.4f}")
    return test_loss


# Training
train_losses = []
test_losses = []
for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    test_loss = test()
    train_losses.append(train_loss)
    test_losses.append(test_loss)


# =================================================================
# VISUALIZATION
# =================================================================
# Visualization: Reconstructions
def visualize_reconstructions():
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:16].to(device)
        recon, _, _, _, _, _, _ = model(data)

        # Plot
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            axes[0, i].imshow(data[i].cpu().reshape(28, 28), cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].imshow(recon[i].cpu().reshape(28, 28), cmap="gray")
            axes[1, i].axis("off")
        plt.suptitle("Original vs Reconstructed")
        plt.show()


# Generate samples with fixed z and varying c
def visualize_latent_space():
    model.eval()
    with torch.no_grad():
        # Fixed universal representation
        z_fixed = torch.randn(1, latent_z).to(device).repeat(8, 1)

        # Varying client representations
        c_vals = torch.randn(8, latent_c).to(device)

        # Generate
        generated = model.client_decoder(z_fixed, c_vals).cpu()

        # Plot
        fig, axes = plt.subplots(1, 8, figsize=(16, 2))
        for i in range(8):
            axes[i].imshow(generated[i].reshape(28, 28), cmap="gray")
            axes[i].axis("off")
        plt.suptitle("Fixed z, Varying c")
        plt.show()

        # Fixed client representation
        c_fixed = torch.randn(1, latent_c).to(device).repeat(8, 1)

        # Varying universal representations
        z_vals = torch.randn(8, latent_z).to(device)

        # Generate
        generated = model.client_decoder(z_vals, c_fixed).cpu()

        # Plot
        fig, axes = plt.subplots(1, 8, figsize=(16, 2))
        for i in range(8):
            axes[i].imshow(generated[i].reshape(28, 28), cmap="gray")
            axes[i].axis("off")
        plt.suptitle("Fixed c, Varying z")
        plt.show()


# Generate samples by interpolating in latent space
def visualize_interpolation():
    model.eval()
    with torch.no_grad():
        # Get two samples
        data, _ = next(iter(test_loader))
        x1, x2 = data[0:1].to(device), data[1:2].to(device)

        # Get their latent representations
        _, z1, c1, _, _, _, _ = model(x1)
        _, z2, c2, _, _, _, _ = model(x2)

        # Interpolation steps
        steps = 8
        interpolations = []
        for alpha in torch.linspace(0, 1, steps):
            z = alpha * z1 + (1 - alpha) * z2
            c = alpha * c1 + (1 - alpha) * c2
            recon = model.client_decoder(z, c)
            interpolations.append(recon.cpu())

        # Plot
        fig, axes = plt.subplots(1, steps, figsize=(16, 2))
        for i in range(steps):
            axes[i].imshow(interpolations[i].reshape(28, 28), cmap="gray")
            axes[i].axis("off")
        plt.suptitle("Latent Space Interpolation")
        plt.show()


# Run visualizations
print("Visualizing reconstructions...")
visualize_reconstructions()

print("Visualizing latent space variations...")
visualize_latent_space()

print("Visualizing latent space interpolation...")
visualize_interpolation()
