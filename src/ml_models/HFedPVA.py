import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ml_models.cnn import CNN
from src.ml_models.decoder import Decoder
from src.ml_models.generic_encoder import GenericEncoder
from src.ml_models.personalized_encoder import PersonalizedEncoder


class HFedPVA(nn.Module):
    def __init__(self, cnn_type: str):
        super().__init__()
        self.cnn_classifier = CNN(cnn_type)
        self.generic_encoder = GenericEncoder()
        self.personalized_encoder = PersonalizedEncoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def combined_loss(
        self,
        y_logits,
        y_true,
        recon_x,
        x,
        mu_z,
        logvar_z,
        mu_c,
        logvar_c,
        all_mu_c,
        all_logvar_c,
    ):
        # Classification loss
        ce_loss = F.cross_entropy(y_logits, y_true, reduction="sum")

        # Reconstruction loss
        BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

        # KL Regularizers (Eq.6-7)
        # R_z: KL(q(z|x) || p(z)) (p(z) = N(0,I))
        KLD_z = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

        # Compute p_k(c) as mixture distribution (Eq.5)
        mu_c_mix = torch.mean(all_mu_c, dim=0)
        logvar_c_mix = torch.mean(all_logvar_c, dim=0)

        # KL(q(c|x,z) || p(c)) (p(c) = N(0,I))
        KLD_c_prior = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())

        # KL(q(c|x,z) || p_k(c))
        var_c = logvar_c.exp()
        var_c_mix = logvar_c_mix.exp()
        KLD_c_client = 0.5 * torch.sum(
            logvar_c_mix - logvar_c + (var_c + (mu_c - mu_c_mix).pow(2)) / var_c_mix - 1
        )

        # Slack regularizer (Eq.5,7)
        R_c = torch.max(0.01 + KLD_c_client, KLD_c_prior)

        # Final loss (Eq.4)
        feddva_loss = BCE + KLD_z + R_c

        total_loss = ce_loss + feddva_loss

        return total_loss, ce_loss, BCE, KLD_z, R_c

    def forward(self, x):
        y_logits = self.cnn_classifier(x)

        # Generic representation
        mu_z, logvar_z = self.generic_encoder(y_logits)
        z = self.reparameterize(mu_z, logvar_z)

        # Personalized representation
        mu_c, logvar_c = self.personalized_encoder(y_logits, z)
        c = self.reparameterize(mu_c, logvar_c)

        # Reconstruction
        recon_x = self.decoder(z, c)

        return y_logits, recon_x, mu_z, logvar_z, mu_c, logvar_c
