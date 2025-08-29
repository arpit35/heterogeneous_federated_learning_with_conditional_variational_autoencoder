import torch
import torch.nn.functional as F

from src.ml_models.utils import to_onehot
from src.scripts.helper import metadata


def decoder_combined_loss(
    images,
    labels,
    logits,
    recon_x,
    mu,
    logvar,
):
    classification_loss = F.cross_entropy(logits, labels, reduction="sum")

    width = metadata["image_width"]
    height = metadata["image_height"]
    # recon_x and x in [0,1]
    # Reconstruction loss (binary cross entropy) summed over pixels
    BCE = F.binary_cross_entropy(
        recon_x.view(-1, width * height),
        images.view(-1, width * height),
        reduction="sum",
    )

    # KL divergence between q(z|x) and N(0,1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return classification_loss + BCE + KLD, classification_loss, BCE, KLD


def train_conditional_vae_with_classification(
    net,
    trainloader,
    testloader,
    epochs,
    learning_rate,
    device,
    dataset_input_feature,
    dataset_target_feature,
    decoderLatentSpace,
    decoder,
):
    """Train the model on the training set."""
    net.to(device)
    decoderLatentSpace.to(device)
    decoder.to(device)

    net_optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    decoderLatentSpace_optimizer = torch.optim.Adam(
        decoderLatentSpace.parameters(), lr=learning_rate
    )
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            net_optimizer.zero_grad()
            decoderLatentSpace_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)
            y_onehot = to_onehot(
                labels, num_classes=metadata["num_classes"], device=device
            )

            logits, in_decoder_latent_space = net(images)
            decoder_latent_z, mu, logvar = decoderLatentSpace(in_decoder_latent_space)
            recon_x = decoder(decoder_latent_z, y_onehot)

            # Calculate loss
            combined_loss, _, _, _ = decoder_combined_loss(
                images,
                labels,
                logits,
                recon_x,
                mu,
                logvar,
            )

            combined_loss.backward()
            net_optimizer.step()
            decoderLatentSpace_optimizer.step()
            decoder_optimizer.step()

        return test_conditional_vae_with_classification(
            net,
            decoderLatentSpace,
            decoder,
            testloader,
            device,
            dataset_input_feature,
            dataset_target_feature,
        )


def test_conditional_vae_with_classification(
    net,
    decoderLatentSpace,
    decoder,
    testloader,
    device,
    dataset_input_feature,
    dataset_target_feature,
):
    """Validate the model on the test set."""
    net.to(device)
    decoderLatentSpace.to(device)
    decoder.to(device)

    net.eval()
    decoderLatentSpace.eval()
    decoder.eval()

    correct = 0
    testloader_length = len(testloader.dataset)
    total_combined_loss = 0.0
    total_classification_loss = 0.0
    total_BCE = 0.0
    total_KLD = 0.0

    with torch.no_grad():
        for batch in testloader:
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)
            y_onehot = to_onehot(
                labels, num_classes=metadata["num_classes"], device=device
            )

            logits, in_decoder_latent_space = net(images)
            decoder_latent_z, mu, logvar = decoderLatentSpace(in_decoder_latent_space)
            recon_x = decoder(decoder_latent_z, y_onehot)

            combined_loss, classification_loss, BCE, KLD = decoder_combined_loss(
                images,
                labels,
                logits,
                recon_x,
                mu,
                logvar,
            )

            correct += (torch.max(logits, 1)[1] == labels).sum().item()

            total_combined_loss += combined_loss.item()
            total_classification_loss += classification_loss.item()
            total_BCE += BCE.item()
            total_KLD += KLD.item()

    avg_classification_loss = total_classification_loss / testloader_length
    avg_BCE = total_BCE / testloader_length
    avg_KLD = total_KLD / testloader_length
    avg_combined_loss = total_combined_loss / testloader_length
    accuracy = correct / testloader_length

    return {
        "combined_loss": avg_combined_loss,
        "accuracy": accuracy,
        "classification_loss": avg_classification_loss,
        "BCE": avg_BCE,
        "KLD": avg_KLD,
    }
