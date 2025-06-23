from collections import OrderedDict

import torch


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(
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
    net.to(device)  # move model to GPU if available
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)
            optimizer.zero_grad()

            y_logits, recon_x, mu_z, logvar_z, mu_c, logvar_c = net(images)

            # Collect statistics for p_k(c) from current batch
            all_mu_c = mu_c.detach().clone()
            all_logvar_c = logvar_c.detach().clone()

            # Calculate loss
            loss, _, _, _ = net.combined_loss(
                y_logits,
                labels,
                recon_x,
                images,
                mu_z,
                logvar_z,
                mu_c,
                logvar_c,
                all_mu_c,
                all_logvar_c,
            )

            loss.backward()
            optimizer.step()

    test_loss, test_acc = test(
        net,
        testloader,
        device,
        dataset_input_feature,
        dataset_target_feature,
    )

    results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }
    return results


def test(net, testloader, device, dataset_input_feature, dataset_target_feature):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)

            y_logits, recon_x, mu_z, logvar_z, mu_c, logvar_c = net(images)

            # Collect statistics for p_k(c) from current batch
            all_mu_c = mu_c.detach().clone()
            all_logvar_c = logvar_c.detach().clone()

            # Calculate loss
            _, ce_loss, _, _ = net.combined_loss(
                y_logits,
                labels,
                recon_x,
                images,
                mu_z,
                logvar_z,
                mu_c,
                logvar_c,
                all_mu_c,
                all_logvar_c,
            )

            correct += (torch.max(y_logits, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = ce_loss / len(testloader.dataset)
    return loss, accuracy
