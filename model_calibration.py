from src.ml_models.cnn import CNN
from src.ml_models.utils import count_params
from src.ml_models.vae import VAE
from src.scripts.helper import save_metadata

cnn = CNN("cnn5")

vae_parameters = {
    "h_dim": 248,
    "res_h_dim": 60,
    "n_res_layers": 3,
    "latent_dim": 100,
}

vae = VAE(**vae_parameters)

print("Cnn Parameters:", count_params(cnn))
print("Vae Parameters:", count_params(vae))

save_metadata({"vae_parameters": vae_parameters})
