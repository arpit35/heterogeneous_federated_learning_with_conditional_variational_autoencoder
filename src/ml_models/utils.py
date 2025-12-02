import os
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from src.scripts.helper import metadata


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def to_onehot(labels, num_classes, device):
    onehot = torch.zeros(labels.size(0), num_classes, device=device)
    onehot.scatter_(1, labels.view(-1, 1), 1)
    return onehot


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def generate_and_save_images(
    vqvae,
    pixel_cnn,
    device,
    output_folder,
    current_round,
    num_classes=10,
    images_per_class=10,
):
    """
    Generate and save images for all classes using VQVAE decoder and PixelCNN.

    Args:
        vqvae: Trained VQVAE model
        pixel_cnn: Trained PixelCNN model
        device: Device to run models on
        output_folder: Folder to save generated images
        current_round: Current training round
        num_classes: Number of classes (default 10 for CIFAR-10)
        images_per_class: Number of images to generate per class
    """
    vqvae.eval()
    pixel_cnn.eval()

    # Create output directory for this round
    round_folder = os.path.join(output_folder, f"round_{current_round}")
    os.makedirs(round_folder, exist_ok=True)

    latent_img_dim = int(np.sqrt(metadata["embedding_dim"]))

    with torch.no_grad():
        for class_label in range(num_classes):
            class_folder = os.path.join(round_folder, f"class_{class_label}")
            os.makedirs(class_folder, exist_ok=True)

            for img_idx in range(images_per_class):
                # Initialize latent image with zeros
                latent_img = torch.zeros(
                    1, latent_img_dim, latent_img_dim, dtype=torch.long, device=device
                )

                # Generate latent representation using PixelCNN
                for i in range(latent_img_dim):
                    for j in range(latent_img_dim):
                        logits = pixel_cnn(
                            latent_img, torch.tensor([class_label], device=device)
                        )
                        logits = logits[:, :, i, j]
                        probs = torch.softmax(logits, dim=-1)
                        latent_img[:, i, j] = torch.multinomial(probs, 1).squeeze(-1)

                # Decode latent representation to image space
                latent_embedding = vqvae.vector_quantization.embedding(
                    latent_img.view(1, -1)
                )
                latent_embedding = latent_embedding.view(
                    1, metadata["embedding_dim"], latent_img_dim, latent_img_dim
                )
                decoded_img = vqvae.decoder(latent_embedding)

                # Normalize image to [0, 255] range
                decoded_img = decoded_img.squeeze(0).cpu()
                decoded_img = (
                    decoded_img + 1
                ) / 2  # Assuming images are normalized to [-1, 1]
                decoded_img = torch.clamp(decoded_img, 0, 1)
                decoded_img = (decoded_img * 255).numpy().astype(np.uint8)

                # Convert to PIL Image and save
                if decoded_img.shape[0] == 3:  # RGB image
                    pil_img = Image.fromarray(
                        decoded_img.transpose(1, 2, 0), mode="RGB"
                    )
                else:  # Grayscale image
                    pil_img = Image.fromarray(decoded_img.squeeze(), mode="L")

                img_path = os.path.join(class_folder, f"image_{img_idx:03d}.png")
                pil_img.save(img_path)
