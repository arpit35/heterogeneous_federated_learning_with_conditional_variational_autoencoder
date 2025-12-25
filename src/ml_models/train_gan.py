import torch
import torch.nn as nn

from src.scripts.helper import metadata


def train_gan(
    generator,
    discriminator,
    trainloader,
    epochs,
    device,
    dataset_input_feature,
):
    """Train VQVAE and PixelCNN sequentially for better convergence."""
    generator.to(device)
    discriminator.to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters())
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters())

    generator.train()
    discriminator.train()

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
            z = torch.randn(
                batch_size,
                metadata["generator_parameters"]["latent_dim"],
                device=device,
            )
            gen_imgs = generator(z)

            fake_pred = discriminator(gen_imgs.detach())
            discriminator_fake_loss = bce_loss(fake_pred, fake)
            # Total discriminator loss
            discriminator_loss = (discriminator_real_loss + discriminator_fake_loss) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Generate images and calculate loss
            generator_optimizer.zero_grad()

            validity = discriminator(gen_imgs)
            generator_loss = bce_loss(validity, valid)
            generator_loss.backward()
            generator_optimizer.step()

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

    del generator_optimizer
    del discriminator_optimizer
    torch.cuda.empty_cache()

    return {
        "loss": avg_loss,
        "discriminator_loss": avg_discriminator_loss,
        "generator_loss": avg_generator_loss,
    }
