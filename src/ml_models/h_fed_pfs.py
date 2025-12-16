import torch.nn as nn
import torch.nn.functional as F

from src.scripts.helper import metadata


class FPN(nn.Module):
    def __init__(self, feature_dim=metadata["f_dim"]):
        super().__init__()
        self.fc = nn.Linear(feature_dim, feature_dim * 2)
        self.ln = nn.LayerNorm(feature_dim * 2)

    def forward(self, h):
        z = self.fc(h)
        z = self.ln(z)
        z = F.relu(z)
        z = z.view(z.size(0), 2, -1)

        # Gumbel-Softmax over {generic, personalized}
        scores = F.gumbel_softmax(z, tau=1.0, hard=False, dim=1)
        sigma = scores[:, 0, :]
        mu = scores[:, 1, :]
        return sigma, mu


# ----------------------------------------------------
# Global Homogeneous Adapter
# ----------------------------------------------------


class Adapter(nn.Module):
    def __init__(
        self, feature_dim=metadata["f_dim"], num_classes=metadata["num_classes"]
    ):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, p):
        return self.fc(p)
