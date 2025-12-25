from src.ml_models.cnn import CNN
from src.ml_models.discriminator import Discriminator
from src.ml_models.generator import Generator
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

generator_parameters = {
    "block_repeat": 2,
    "n_block_layers": 6,
    "h_dim": 220,
    "latent_dim": 100,
}
discriminator_parameters = {
    "block_repeat": 2,
    "n_block_layers": 4,
    "h_dim": 256,
}

generator = Generator(**generator_parameters)
discriminator = Discriminator(**discriminator_parameters)

generator_parameters_count = count_params(generator)
discriminator_parameters_count = count_params(discriminator)

print("Generator Parameters:", generator_parameters_count)
print("Discriminator Parameters:", discriminator_parameters_count)
print(
    "Total GAN Parameters:",
    generator_parameters_count + discriminator_parameters_count,
)

save_metadata(
    {
        "vae_parameters": vae_parameters,
        "generator_parameters": generator_parameters,
        "discriminator_parameters": discriminator_parameters,
    }
)
