import torch
import torch.nn.functional as F

from src.ml_models.train_net import test_net


def gaussian_mmd(x, y, sigma=1.0):
    """Simple MMD with Gaussian kernel."""
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    Kxx = torch.exp(-(rx.t() + rx - 2 * xx) / (2 * sigma**2))
    Kyy = torch.exp(-(ry.t() + ry - 2 * yy) / (2 * sigma**2))
    Kxy = torch.exp(-(rx.t() + ry - 2 * xy) / (2 * sigma**2))

    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def train_h_fed_pfs(
    net,
    fpn,
    adapter,
    trainloader,
    testloader,
    epochs,
    learning_rate,
    device,
    dataset_input_feature,
    dataset_target_feature,
    optimizer_strategy="sgd",
):
    net.to(device)
    fpn.to(device)
    adapter.to(device)

    if optimizer_strategy == "sgd":
        optimizer_net = torch.optim.SGD(net.parameters(), lr=learning_rate)
        optimizer_fpn = torch.optim.SGD(fpn.parameters(), lr=learning_rate)
        optimizer_adapter = torch.optim.SGD(adapter.parameters(), lr=learning_rate)
    elif optimizer_strategy == "adam":
        optimizer_net = torch.optim.Adam(net.parameters(), lr=learning_rate)
        optimizer_fpn = torch.optim.Adam(fpn.parameters(), lr=learning_rate)
        optimizer_adapter = torch.optim.Adam(adapter.parameters(), lr=learning_rate)

    net.train()
    fpn.train()
    adapter.train()

    total_loss = 0
    total_L1_loss = 0
    total_L2_loss = 0
    total_L3_loss = 0
    total_samples = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)

            optimizer_net.zero_grad()
            optimizer_fpn.zero_grad()
            optimizer_adapter.zero_grad()

            logits, h = net(images)
            L1 = F.cross_entropy(logits, labels)

            # FPN separation
            sigma, mu = fpn(h)
            g = h * sigma
            p = h * mu

            # Adapter loss
            logits_a = adapter(p)
            L2 = F.cross_entropy(logits_a, labels)

            # MMD regularization
            L3 = gaussian_mmd(sigma, mu)

            loss = L1 + L2 - 0.65 * L3

            loss.backward()

            optimizer_net.step()
            optimizer_fpn.step()
            optimizer_adapter.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_L1_loss += L1.item() * batch_size
            total_L2_loss += L2.item() * batch_size
            total_L3_loss += L3.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_L1_loss = total_L1_loss / total_samples if total_samples > 0 else 0
    avg_L2_loss = total_L2_loss / total_samples if total_samples > 0 else 0
    avg_L3_loss = total_L3_loss / total_samples if total_samples > 0 else 0

    train_loss, train_acc = test_net(
        net, testloader, device, dataset_input_feature, dataset_target_feature
    )

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "total_loss": avg_loss,
        "L1_loss": avg_L1_loss,
        "L2_loss": avg_L2_loss,
        "L3_loss": avg_L3_loss,
    }

    del optimizer_net
    del optimizer_fpn
    del optimizer_adapter
    torch.cuda.empty_cache()

    return results
