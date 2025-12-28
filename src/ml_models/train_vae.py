import torch


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

    for epoch in range(epochs):
        for batch in trainloader:
            vae_optimizer.zero_grad()

            images = batch[dataset_input_feature].to(device)

            recon_x, mu, logvar = vae(images)

            loss, recon_loss, kl_loss = vae.vae_loss(recon_x, images, mu, logvar, epoch)

            loss.backward()
            vae_optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_recon_loss = total_recon_loss / total_samples
    avg_kl_loss = total_kl_loss / total_samples

    del vae_optimizer
    torch.cuda.empty_cache()

    return {
        "loss": avg_loss,
        "recon_loss": avg_recon_loss,
        "kl_loss": avg_kl_loss,
    }
