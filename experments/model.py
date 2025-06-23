import torch
import torch.nn as nn


# Define a small model
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


if __name__ == "__main__":
    net = SmallNet()
    weights_list = get_weights(net)
    print(weights_list)
