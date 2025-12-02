import torch.nn as nn

from src.ml_models.decoder import Decoder
from src.ml_models.encoder import Encoder
from src.ml_models.quantizer import VectorQuantizer
from src.scripts.helper import metadata


class VQVAE(nn.Module):
    def __init__(
        self,
        h_dim=metadata["h_dim"],
        res_h_dim=metadata["res_h_dim"],
        n_res_layers=metadata["n_res_layers"],
        n_embeddings=metadata["n_embeddings"],
        embedding_dim=metadata["embedding_dim"],
        beta=0.25,
        save_img_embedding_map=False,
    ):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1
        )
        self.label_embedding = nn.Embedding(10, embedding_dim)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, min_encoding_indices = (
            self.vector_quantization(z_e)
        )

        x_hat = self.decoder(z_q)

        return embedding_loss, x_hat, perplexity, min_encoding_indices
