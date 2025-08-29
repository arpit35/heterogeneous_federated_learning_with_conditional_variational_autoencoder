from collections import OrderedDict

import torch
import torch.nn.functional as F

from src.scripts.helper import metadata


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def to_onehot(labels, num_classes, device):
    onehot = torch.zeros(labels.size(0), num_classes, device=device)
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot


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

    combined_loss, accuracy, classification_loss, BCE, KLD = (
        test_conditional_vae_with_classification(
            net,
            decoderLatentSpace,
            decoder,
            testloader,
            device,
            dataset_input_feature,
            dataset_target_feature,
        )
    )

    return {
        "combined_loss": combined_loss,
        "accuracy": accuracy,
        "classification_loss": classification_loss,
        "BCE": BCE,
        "KLD": KLD,
    }


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
    accuracy = correct / testloader_length
    combined_loss = combined_loss.item() / testloader_length
    classification_loss = classification_loss.item() / testloader_length

    BCE = BCE.item() / testloader_length
    KLD = KLD.item() / testloader_length

    return combined_loss, accuracy, classification_loss, BCE, KLD


def train_classification(
    net,
    trainloader,
    testloader,
    epochs,
    learning_rate,
    device,
    dataset_input_feature,
    dataset_target_feature,
):
    """Train the model on the training set."""
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            optimizer.zero_grad()

            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)

            logits, _ = net(images)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

    loss, accuracy = test_classification(
        net, testloader, device, dataset_input_feature, dataset_target_feature
    )

    return {"loss": loss, "accuracy": accuracy}


def test_classification(
    net,
    testloader,
    device,
    dataset_input_feature,
    dataset_target_feature,
):
    """Test the model on the test set."""
    net.to(device)
    net.eval()

    correct = 0
    testloader_length = len(testloader.dataset)

    with torch.no_grad():
        for batch in testloader:
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)

            logits, _ = net(images)

            loss = F.cross_entropy(logits, labels)
            correct += (torch.max(logits, 1)[1] == labels).sum().item()

    accuracy = correct / testloader_length
    return loss, accuracy
