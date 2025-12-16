import torch


def train_net(
    net,
    trainloader,
    testloader,
    epochs,
    learning_rate,
    device,
    dataset_input_feature,
    dataset_target_feature,
    optimizer_strategy,
):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if optimizer_strategy == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    elif optimizer_strategy == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    net.train()

    for _ in range(epochs):
        for batch in trainloader:
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)
            optimizer.zero_grad()
            logits, _ = net(images)
            criterion(logits, labels).backward()
            optimizer.step()

    train_loss, train_acc = test_net(
        net, testloader, device, dataset_input_feature, dataset_target_feature
    )

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
    }
    return results


def test_net(
    net,
    testloader,
    device,
    dataset_input_feature,
    dataset_target_feature,
):
    """Validate the model on the test set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)
            logits, _ = net(images)
            loss += criterion(logits, labels).item()
            correct += (torch.max(logits.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader.dataset)
    return loss, accuracy
