import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_dim = 784  # 28x28 flattened
hidden_dim = 400
latent_digit = 30  # Latent space for digit
latent_circle = 15  # Latent space for circle
batch_size = 128
epochs = 20
learning_rate = 0.001


# Function to add random circle to images
def add_circle(img_tensor):
    """Add a random black circle outline to the image tensor"""
    img = img_tensor.clone().squeeze(0)  # Remove channel dim for processing
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

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# =============================================
# VISUALIZE ADDED CIRCLES
# =============================================
def visualize_circles(data_loader, num_images=64):
    """Visualize images with added circles"""
    # Get a batch of images
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    # Reshape images to 28x28
    images = images.view(-1, 28, 28).numpy()

    # Create plot
    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(images))):
        plt.subplot(8, 8, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")
        plt.title(f"L:{labels[i].item()}", fontsize=8)
    plt.suptitle("MNIST Images with Added Circle Outlines", fontsize=16)
    plt.tight_layout()
    plt.show()


# Visualize training data with circles
print("Visualizing training data with added circle outlines...")
visualize_circles(train_loader)


# Dual-Encoder VAE Model
class DualEncoderVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_digit, latent_circle):
        super(DualEncoderVAE, self).__init__()

        # Shared initial encoder layers
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Digit-specific encoder
        self.digit_mu = nn.Linear(hidden_dim, latent_digit)
        self.digit_logvar = nn.Linear(hidden_dim, latent_digit)

        # Circle-specific encoder
        self.circle_mu = nn.Linear(hidden_dim, latent_circle)
        self.circle_logvar = nn.Linear(hidden_dim, latent_circle)

        # Decoder (combines both latent spaces)
        self.decoder = nn.Sequential(
            nn.Linear(latent_digit + latent_circle, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        shared = self.shared_encoder(x)
        digit_mu = self.digit_mu(shared)
        digit_logvar = self.digit_logvar(shared)
        circle_mu = self.circle_mu(shared)
        circle_logvar = self.circle_logvar(shared)
        return digit_mu, digit_logvar, circle_mu, circle_logvar

    def decode(self, digit_z, circle_z):
        z_combined = torch.cat([digit_z, circle_z], dim=1)
        return self.decoder(z_combined)

    def forward(self, x):
        # Encoding
        digit_mu, digit_logvar, circle_mu, circle_logvar = self.encode(x)

        # Reparameterization
        digit_z = self.reparameterize(digit_mu, digit_logvar)
        circle_z = self.reparameterize(circle_mu, circle_logvar)

        # Decoding
        x_recon = self.decode(digit_z, circle_z)

        return x_recon, digit_mu, digit_logvar, circle_mu, circle_logvar


# Loss function
def loss_function(recon_x, x, mu_d, logvar_d, mu_c, logvar_c):
    # Reconstruction loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")

    # KL Divergences
    KLD_digit = -0.5 * torch.sum(1 + logvar_d - mu_d.pow(2) - logvar_d.exp())
    KLD_circle = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())

    return BCE + KLD_digit + KLD_circle


# Initialize model
model = DualEncoderVAE(input_dim, hidden_dim, latent_digit, latent_circle).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu_d, logvar_d, mu_c, logvar_c = model(data)
        loss = loss_function(recon_batch, data, mu_d, logvar_d, mu_c, logvar_c)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}"
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
            recon, mu_d, logvar_d, mu_c, logvar_c = model(data)
            test_loss += loss_function(
                recon, data, mu_d, logvar_d, mu_c, logvar_c
            ).item()

    test_loss /= len(test_loader.dataset)
    print(f"====> Test set loss: {test_loss:.4f}")
    return test_loss


# Training and testing
train_losses = []
test_losses = []
for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    test_loss = test()
    train_losses.append(train_loss)
    test_losses.append(test_loss)

# Visualization: Generate samples with fixed circles and varying digits
with torch.no_grad():
    # Generate random digit latents
    digit_z = torch.randn(64, latent_digit).to(device)

    # Create fixed circle latents (one per row)
    circle_z = torch.zeros(64, latent_circle).to(device)
    for i in range(8):
        circle_sample = torch.randn(1, latent_circle).to(device)
        circle_z[i * 8 : (i + 1) * 8] = circle_sample.repeat(8, 1)

    # Combine and decode
    generated = model.decode(digit_z, circle_z).cpu()

# Visualize generated images
fig, ax = plt.subplots(8, 8, figsize=(10, 10))
for i in range(64):
    ax[i // 8, i % 8].imshow(generated[i].view(28, 28), cmap="gray")
    ax[i // 8, i % 8].axis("off")
plt.suptitle("Generated Images: Fixed Circles per Row, Varying Digits", fontsize=16)
plt.show()

# Visualization: Generate samples with fixed digits and varying circles
with torch.no_grad():
    # Create fixed digit latents (one per column)
    digit_z = torch.zeros(64, latent_digit).to(device)
    for i in range(8):
        digit_sample = torch.randn(1, latent_digit).to(device)
        digit_z[i:64:8] = digit_sample.repeat(8, 1)

    # Generate random circle latents
    circle_z = torch.randn(64, latent_circle).to(device)

    # Combine and decode
    generated = model.decode(digit_z, circle_z).cpu()

# Visualize generated images
fig, ax = plt.subplots(8, 8, figsize=(10, 10))
for i in range(64):
    ax[i // 8, i % 8].imshow(generated[i].view(28, 28), cmap="gray")
    ax[i // 8, i % 8].axis("off")
plt.suptitle("Generated Images: Fixed Digits per Column, Varying Circles", fontsize=16)
plt.show()
