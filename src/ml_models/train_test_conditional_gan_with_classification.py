import torch
import torch.nn.functional as F

from src.ml_models.utils import to_onehot
from src.scripts.helper import metadata


def train_conditional_gan_with_classification(
    net,
    trainloader,
    testloader,
    epochs,
    learning_rate,
    device,
    dataset_input_feature,
    dataset_target_feature,
    decoder,
):
    """Train the model on the training set."""
    net.to(device)
    decoder.to(device)

    net_optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            net_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)
            y_onehot = to_onehot(
                labels, num_classes=metadata["num_classes"], device=device
            )
            decoder_latent_z = torch.randn(
                labels.size(0), metadata["decoder_latent_dim"], device=device
            )

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Real images
            x_pred, _ = net(images)
            classification_loss = F.cross_entropy(x_pred, labels)

            # gen images
            gen_x = decoder(decoder_latent_z, y_onehot)
            gen_x_pred, _ = net(gen_x.detach())
            gen_x_loss = -F.cross_entropy(gen_x_pred, labels)

            # Total discriminator loss
            d_loss = (classification_loss + gen_x_loss) / 2
            d_loss.backward()
            net_optimizer.step()

            # -----------------
            #  Train Generator
            # -----------------

            gen_x_pred, _ = net(gen_x)
            gen_x_loss = F.cross_entropy(gen_x_pred, labels)

            # generator loss
            gen_x_loss.backward()
            decoder_optimizer.step()

    return test_conditional_gan_with_classification(
        net,
        decoder,
        testloader,
        device,
        dataset_input_feature,
        dataset_target_feature,
    )


def test_conditional_gan_with_classification(
    net,
    decoder,
    testloader,
    device,
    dataset_input_feature,
    dataset_target_feature,
):
    """Validate the model on the test set."""
    net.to(device)
    decoder.to(device)

    net.eval()
    decoder.eval()

    correct = 0
    testloader_length = len(testloader.dataset)
    total_classification_loss = 0.0
    total_gen_x_discriminator_loss = 0.0
    total_gen_x_generator_loss = 0.0

    with torch.no_grad():
        for batch in testloader:
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)
            y_onehot = to_onehot(
                labels, num_classes=metadata["num_classes"], device=device
            )
            decoder_latent_z = torch.randn(
                labels.size(0), metadata["decoder_latent_dim"], device=device
            )

            # Real images
            x_pred, _ = net(images)
            classification_loss = F.cross_entropy(x_pred, labels)

            # gen images
            gen_x = decoder(decoder_latent_z, y_onehot)
            gen_x_pred, _ = net(gen_x)
            gen_x_discriminator_loss = -F.cross_entropy(gen_x_pred, labels)
            gen_x_generator_loss = F.cross_entropy(gen_x_pred, labels)

            correct += (torch.max(x_pred, 1)[1] == labels).sum().item()

            # Accumulate losses
            total_classification_loss += classification_loss.item()
            total_gen_x_discriminator_loss += gen_x_discriminator_loss.item()
            total_gen_x_generator_loss += gen_x_generator_loss.item()

    # Calculate average losses
    avg_classification_loss = total_classification_loss / testloader_length
    avg_gen_x_discriminator_loss = total_gen_x_discriminator_loss / testloader_length
    avg_gen_x_generator_loss = total_gen_x_generator_loss / testloader_length

    combined_loss = (
        avg_classification_loss
        + avg_gen_x_discriminator_loss
        + avg_gen_x_generator_loss
    )
    accuracy = correct / testloader_length

    return {
        "combined_loss": combined_loss,
        "accuracy": accuracy,
        "classification_loss": avg_classification_loss,
        "gen_x_discriminator_loss": avg_gen_x_discriminator_loss,
        "gen_x_generator_loss": avg_gen_x_generator_loss,
    }
