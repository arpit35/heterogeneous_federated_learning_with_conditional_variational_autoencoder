import torch
import torch.nn as nn

from src.scripts.helper import metadata


def train_vae(
    vae,
    discriminator,
    trainloader,
    epochs,
    device,
    dataset_input_feature,
):
    """Train VQVAE and PixelCNN sequentially for better convergence."""
    # vae.fc_expand.to(device)
    # vae.decoder.to(device)
    discriminator.to(device)
    vae.to(device)

    # vae_fc_expand_optimizer = torch.optim.Adam(vae.fc_expand.parameters())
    # vae_decoder_optimizer = torch.optim.Adam(vae.decoder.parameters())
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters())
    vae_optimizer = torch.optim.Adam(vae.parameters())

    # vae.fc_expand.train()
    # vae.decoder.train()
    discriminator.train()
    vae.train()

    total_loss = 0
    total_discriminator_loss = 0
    total_generator_loss = 0
    total_samples = 0

    bce_loss = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        for batch in trainloader:
            images = batch[dataset_input_feature].to(device)
            batch_size = images.size(0)

            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            discriminator_optimizer.zero_grad()

            # Real images
            real_pred = discriminator(images)
            discriminator_real_loss = bce_loss(real_pred, valid)

            # Fake images
            # z = torch.randn(batch_size, metadata["latent_dim"], device=device)
            z = torch.randn(batch_size, 100, device=device)
            # expanded = vae.fc_expand(z)
            # expanded = expanded.view(batch_size, *vae.enc_shape)
            # gen_imgs = vae.decoder(expanded)
            gen_imgs = vae(z)

            fake_pred = discriminator(gen_imgs.detach())
            discriminator_fake_loss = bce_loss(fake_pred, fake)
            # Total discriminator loss
            discriminator_loss = (discriminator_real_loss + discriminator_fake_loss) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Generate images and calculate loss
            # vae_fc_expand_optimizer.zero_grad()
            # vae_decoder_optimizer.zero_grad()
            vae_optimizer.zero_grad()

            validity = discriminator(gen_imgs)
            generator_loss = bce_loss(validity, valid)
            generator_loss.backward()
            # vae_fc_expand_optimizer.step()
            # vae_decoder_optimizer.step()
            vae_optimizer.step()

            total_discriminator_loss += discriminator_loss.item() * batch_size
            total_generator_loss += generator_loss.item() * batch_size
            total_loss += total_discriminator_loss + total_generator_loss
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_discriminator_loss = (
        total_discriminator_loss / total_samples if total_samples > 0 else 0
    )
    avg_generator_loss = (
        total_generator_loss / total_samples if total_samples > 0 else 0
    )

    # del vae_fc_expand_optimizer
    # del vae_decoder_optimizer
    del discriminator_optimizer
    del vae_optimizer
    torch.cuda.empty_cache()

    return {
        "loss": avg_loss,
        "discriminator_loss": avg_discriminator_loss,
        "generator_loss": avg_generator_loss,
    }
