import torch.nn as nn
import torch.nn.functional as F

from src.scripts.helper import metadata

cnn_config = {
    "cnn1": {
        "conv1_filters": 16,
        "conv2_filters": 32,
        "conv1_kernel_size": 5,
        "conv2_kernel_size": 5,
        "fc1_out_features": 2000,
        "fc2_out_features": 500,
    },
    "cnn2": {
        "conv1_filters": 16,
        "conv2_filters": 16,
        "conv1_kernel_size": 5,
        "conv2_kernel_size": 5,
        "fc1_out_features": 2000,
        "fc2_out_features": 500,
    },
    "cnn3": {
        "conv1_filters": 16,
        "conv2_filters": 32,
        "conv1_kernel_size": 5,
        "conv2_kernel_size": 5,
        "fc1_out_features": 1000,
        "fc2_out_features": 500,
    },
    "cnn4": {
        "conv1_filters": 16,
        "conv2_filters": 32,
        "conv1_kernel_size": 5,
        "conv2_kernel_size": 5,
        "fc1_out_features": 800,
        "fc2_out_features": 500,
    },
    "cnn5": {
        "conv1_filters": 16,
        "conv2_filters": 32,
        "conv1_kernel_size": 5,
        "conv2_kernel_size": 5,
        "fc1_out_features": 500,
        "fc2_out_features": 500,
    },
}


class CNN(nn.Module):
    def __init__(self, cnn_type: str):
        super().__init__()
        (
            conv1_filters,
            conv2_filters,
            conv1_kernel_size,
            conv2_kernel_size,
            fc1_out_features,
            fc2_out_features,
        ) = cnn_config[cnn_type].values()

        # Input: 1x32x32
        self.conv1 = nn.Conv2d(
            in_channels=metadata["num_channels"],
            out_channels=conv1_filters,
            kernel_size=conv1_kernel_size,
        )  # → 16x28x28
        self.conv2 = nn.Conv2d(
            in_channels=conv1_filters,
            out_channels=conv2_filters,
            kernel_size=conv2_kernel_size,
        )  # → 32x24x24

        # Flatten size: 32 × 24 × 24 = 18432
        self.fc1 = nn.Linear(
            conv2_filters
            * (metadata["image_height"] - conv1_kernel_size + 1 - conv2_kernel_size + 1)
            * (metadata["image_width"] - conv1_kernel_size + 1 - conv2_kernel_size + 1),
            fc1_out_features,
        )
        self.fc2 = nn.Linear(fc1_out_features, fc2_out_features)
        self.fc3 = nn.Linear(
            fc2_out_features, metadata["num_classes"]
        )  # Output: num_classes

    def forward(self, x):
        x = F.relu(self.conv1(x))  # → 16x28x28
        x = F.relu(self.conv2(x))  # → 32x24x24
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))  # → 2000
        x = F.relu(self.fc2(x))  # → 500
        out = self.fc3(x)  # → num_classes

        return out, x
