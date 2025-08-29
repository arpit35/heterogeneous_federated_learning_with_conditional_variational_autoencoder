import torch
import torch.nn.functional as F


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

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

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

    return test_classification(
        net, testloader, device, dataset_input_feature, dataset_target_feature
    )


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
    total_loss = 0.0

    with torch.no_grad():
        for batch in testloader:
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)

            logits, _ = net(images)

            loss = F.cross_entropy(logits, labels)
            correct += (torch.max(logits, 1)[1] == labels).sum().item()

            total_loss += loss.item()

    avg_loss = total_loss / testloader_length
    accuracy = correct / testloader_length

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
    }
