from collections import OrderedDict

import torch

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


def train(
    net,
    trainloader,
    testloader,
    epochs,
    learning_rate,
    device,
    dataset_input_feature,
    dataset_target_feature,
    detach_decoder=False,
):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)
            y_onehot = to_onehot(
                labels, num_classes=metadata["num_classes"], device=device
            )

            optimizer.zero_grad()

            logits, recon_x, mu, logvar = net(images, y_onehot, detach_decoder)

            # Calculate loss
            combined_loss, _, _, _ = net.combined_loss(
                images,
                labels,
                logits,
                recon_x,
                mu,
                logvar,
            )

            combined_loss.backward()
            optimizer.step()

    combined_loss, accuracy, classification_loss, BCE, KLD = test(
        net,
        testloader,
        device,
        dataset_input_feature,
        dataset_target_feature,
        detach_decoder,
    )

    results = {
        "combined_loss": combined_loss,
        "accuracy": accuracy,
        "classification_loss": classification_loss,
        "BCE": BCE,
        "KLD": KLD,
    }
    return results


def test(
    net,
    testloader,
    device,
    dataset_input_feature,
    dataset_target_feature,
    detach_decoder=False,
):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    correct = 0
    testloader_length = len(testloader.dataset)

    with torch.no_grad():
        for batch in testloader:
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)
            y_onehot = to_onehot(
                labels, num_classes=metadata["num_classes"], device=device
            )

            logits, recon_x, mu, logvar = net(images, y_onehot, detach_decoder)

            combined_loss, classification_loss, BCE, KLD = net.combined_loss(
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

    if not detach_decoder:
        BCE = BCE.item() / testloader_length
        KLD = KLD.item() / testloader_length

    return combined_loss, accuracy, classification_loss, BCE, KLD
