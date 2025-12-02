import torch
import torch.nn as nn

from src.scripts.helper import metadata


def train_vqvae_with_pixel_cnn(
    vqvae,
    pixel_cnn,
    trainloader,
    epochs,
    device,
    dataset_input_feature,
    dataset_target_feature,
    x_train_var,
):
    """Train the model on the training set."""
    vqvae.to(device)
    pixel_cnn.to(device)

    vqvae_optimizer = torch.optim.Adam(vqvae.parameters())
    pixel_cnn_optimizer = torch.optim.Adam(pixel_cnn.parameters())

    pixel_cnn_criterion = nn.CrossEntropyLoss().cuda()

    vqvae.train()
    pixel_cnn.train()

    total_loss = 0
    total_recon_loss = 0
    total_perplexity = 0
    total_pixel_cnn_loss = 0
    total_samples = 0

    for _ in range(epochs):
        for batch in trainloader:
            vqvae_optimizer.zero_grad()
            pixel_cnn_optimizer.zero_grad()

            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)

            embedding_loss, x_hat, perplexity, min_encoding_indices = vqvae(images)
            recon_loss = torch.mean((x_hat - images) ** 2) / x_train_var
            loss = recon_loss + embedding_loss

            loss.backward()
            vqvae_optimizer.step()

            batch_size = images.size(0)

            latent_img_dim = int(torch.sqrt(torch.tensor(metadata["embedding_dim"])))
            latent_img = (
                min_encoding_indices.view(batch_size, latent_img_dim, latent_img_dim)
                .squeeze(-1)
                .to(device)
            )

            logits = pixel_cnn(latent_img, labels)
            logits = logits.permute(0, 2, 3, 1).contiguous()

            pixel_cnn_loss = pixel_cnn_criterion(
                logits.view(-1, metadata["n_embeddings"]), latent_img.view(-1)
            )

            pixel_cnn_loss.backward()
            pixel_cnn_optimizer.step()

            total_loss += loss.item() * batch_size
            total_recon_loss += recon_loss.item() * batch_size
            total_perplexity += perplexity.item() * batch_size
            total_pixel_cnn_loss += pixel_cnn_loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_recon_loss = total_recon_loss / total_samples if total_samples > 0 else 0
    avg_perplexity = total_perplexity / total_samples if total_samples > 0 else 0
    avg_pixel_cnn_loss = (
        total_pixel_cnn_loss / total_samples if total_samples > 0 else 0
    )

    return {
        "loss": avg_loss,
        "recon_loss": avg_recon_loss,
        "perplexity": avg_perplexity,
        "pixel_cnn_loss": avg_pixel_cnn_loss,
    }
