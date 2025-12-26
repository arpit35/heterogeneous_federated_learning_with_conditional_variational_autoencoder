import torch

# from experments.gan import Discriminator, Generator
from src.ml_models.cnn import CNN
from src.ml_models.discriminator import Discriminator
from src.ml_models.generator import Generator
from src.ml_models.utils import count_params
from src.ml_models.vae import VAE
from src.scripts.helper import save_metadata

save_model_path = "model_architecture/"

cnn = CNN("cnn5")

cnn_dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(cnn, cnn_dummy_input, save_model_path + "cnn.onnx")

vae_parameters = {
    "h_dim": 248,
    "res_h_dim": 60,
    "n_res_layers": 3,
    "latent_dim": 100,
}

vae = VAE(**vae_parameters)

vae_dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(vae, vae_dummy_input, save_model_path + "vae.onnx")

print("Cnn Parameters:", count_params(cnn))
print("Vae Parameters:", count_params(vae))

generator_parameters = {
    "n_block_layers": 2,
    "h_dim": 187,
    "latent_dim": 100,
    "init_img_dim": 7,
}
discriminator_parameters = {
    "block_repeat": 1,
    "n_block_layers": 4,
    "h_dim": 27,
}

generator = Generator(**generator_parameters)
discriminator = Discriminator(**discriminator_parameters)
# generator = Generator()
# discriminator = Discriminator()

generator_dummy_input = torch.randn(1, 100)
torch.onnx.export(generator, generator_dummy_input, save_model_path + "generator1.onnx")
discriminator_dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    discriminator,
    discriminator_dummy_input,
    save_model_path + "discriminator1.onnx",
)

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
