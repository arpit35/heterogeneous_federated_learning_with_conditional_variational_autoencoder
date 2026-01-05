import torch


def train_vae(
    cvae,
    trainloader,
    epochs,
    device,
    dataset_input_feature,
    dataset_target_feature,
):
    """Train VQVAE and PixelCNN sequentially for better convergence."""
    cvae.to(device)

    cvae_optimizer = torch.optim.Adam(cvae.parameters())

    cvae.train()

    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_samples = 0

    for epoch in range(epochs):
        for batch in trainloader:
            cvae_optimizer.zero_grad()

            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)

            recon_x, mu, logvar = cvae(images, labels)

            loss, recon_loss, kl_loss = cvae.vae_loss(
                recon_x, images, mu, logvar, epoch
            )

            loss.backward()
            cvae_optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_recon_loss = total_recon_loss / total_samples
    avg_kl_loss = total_kl_loss / total_samples

    del cvae_optimizer
    torch.cuda.empty_cache()

    return {
        "loss": avg_loss,
        "recon_loss": avg_recon_loss,
        "kl_loss": avg_kl_loss,
    }
