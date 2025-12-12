import torch
import torch.nn as nn


def vae_loss(recon_x, x, mu, logvar):
    """
    recon_x: reconstructed batch  [B, C, H, W]
    x:       original batch        [B, C, H, W]
    mu:      mean of q(z|x,y)
    logvar:  log variance of q(z|x,y)
    """

    # BCE reconstruction loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")

    # KL divergence term
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss, recon_loss, kl_loss


def train_vae(
    vae,
    trainloader,
    epochs,
    device,
    dataset_input_feature,
):
    """Train VQVAE and PixelCNN sequentially for better convergence."""
    vae.to(device)

    vae_optimizer = torch.optim.Adam(vae.parameters())

    vae.train()

    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_samples = 0

    for _ in range(epochs):
        for batch in trainloader:
            vae_optimizer.zero_grad()

            images = batch[dataset_input_feature].to(device)

            recon_x, mu, logvar = vae(images)

            loss, recon_loss, kl_loss = vae_loss(recon_x, images, mu, logvar)

            loss.backward()
            vae_optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_recon_loss += recon_loss.item() * batch_size
            total_kl_loss += kl_loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_recon_loss = total_recon_loss / total_samples if total_samples > 0 else 0
    avg_kl_loss = total_kl_loss / total_samples if total_samples > 0 else 0

    return {
        "loss": avg_loss,
        "recon_loss": avg_recon_loss,
        "kl_loss": avg_kl_loss,
    }
