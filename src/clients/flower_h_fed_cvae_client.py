import os

import torch
from flwr.client import NumPyClient
from PIL import Image

from src.data_loader import DataLoader
from src.ml_models.cnn import CNN
from src.ml_models.discriminator import Discriminator
from src.ml_models.generator import Generator
from src.ml_models.train_gan import train_gan
from src.ml_models.train_net import test_net, train_net
from src.ml_models.train_vae import train_vae
from src.ml_models.utils import (
    create_synthetic_data,
    get_device,
    get_weights,
    set_weights,
)
from src.ml_models.vae import VAE
from src.scripts.helper import metadata
from src.utils.logger import get_logger


class FlowerHFedCVAEClient(NumPyClient):
    def __init__(
        self,
        client_number,
        batch_size,
        net_epochs,
        vae_epochs,
        net_learning_rate,
        dataset_folder_path,
        model_folder_path,
        dataset_input_feature,
        dataset_target_feature,
        cnn_type,
        mode,
    ):
        super().__init__()
        self.client_number = client_number
        self.batch_size = batch_size
        self.net_epochs = net_epochs
        self.vae_epochs = vae_epochs
        self.net_learning_rate = net_learning_rate
        self.dataset_folder_path = dataset_folder_path
        self.client_data_folder_path = os.path.join(
            dataset_folder_path, f"client_{client_number}"
        )
        self.client_model_folder_path = os.path.join(
            model_folder_path, f"client_{client_number}"
        )
        self.client_metadata_path = os.path.join(
            self.client_data_folder_path, "metadata.json"
        )
        self.dataset_input_feature = dataset_input_feature
        self.dataset_target_feature = dataset_target_feature
        self.cnn_type = cnn_type
        self.mode = mode

        self.net = None
        self.vae = None
        self.generator = None
        self.discriminator = None

        self.device = get_device()

        # Configure logging
        self.logger = get_logger(f"{__name__}_Client_{client_number}", client_number)
        self.logger.info("Client %s initiated", self.client_number)

    def _set_weights_from_disk(self, model):
        if model == "net":
            self.net.load_state_dict(
                torch.load(
                    self.client_model_folder_path + "/model.pth",
                    map_location="cpu",
                )
            )

    def _save_synthetic_images(self, dataloader, current_round):
        """
        Save synthetic images as PNG files in the specified directory structure.

        Structure: {client_model_folder_path}/round_{round}/{label}/{image_index}.png
        """

        # Create base directory for this round
        round_dir = os.path.join(
            self.client_model_folder_path, f"round_{current_round}"
        )
        os.makedirs(round_dir, exist_ok=True)

        image_counter = {}

        # Iterate through the dataloader
        for batch_idx, batch in enumerate(dataloader):
            images = batch[self.dataset_input_feature]
            labels = batch[self.dataset_target_feature]

            # Convert labels to list if they're tensors
            if torch.is_tensor(labels):
                labels = labels.tolist()
            elif not isinstance(labels, list):
                labels = [labels]

            # Process each image in the batch
            for i, (image, label) in enumerate(zip(images, labels)):
                # Initialize counter for this label if not exists
                if label not in image_counter:
                    image_counter[label] = 0
                    # Create label directory
                    label_dir = os.path.join(round_dir, str(label))
                    os.makedirs(label_dir, exist_ok=True)

                # Convert tensor to numpy array for saving
                if torch.is_tensor(image):
                    # Remove batch dimension if present
                    if image.dim() == 4:
                        image = image.squeeze(0)

                    # Convert to PIL Image
                    # Assuming image is in [C, H, W] format and values are in [0, 1]
                    # If values are in [-1, 1], adjust to [0, 1]
                    if image.min() < 0:
                        image = (image + 1) / 2  # Normalize from [-1, 1] to [0, 1]

                    # Convert to numpy and transpose to [H, W, C]
                    image_np = image.cpu().numpy()
                    if image_np.shape[0] == 1:  # Grayscale
                        image_np = image_np.squeeze(0)
                    else:  # RGB or other channels
                        image_np = image_np.transpose(1, 2, 0)

                    # Scale to [0, 255] if needed
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype("uint8")

                    # Create PIL Image
                    if len(image_np.shape) == 2:  # Grayscale
                        pil_image = Image.fromarray(image_np, mode="L")
                    else:  # RGB
                        pil_image = Image.fromarray(image_np, mode="RGB")
                else:
                    # If image is already a numpy array
                    pil_image = Image.fromarray(image.astype("uint8"))

                # Save the image
                image_path = os.path.join(
                    round_dir, str(label), f"{image_counter[label]}.png"
                )
                pil_image.save(image_path)

                # Increment counter for this label
                image_counter[label] += 1

        self.logger.info(
            f"Saved {sum(image_counter.values())} synthetic images to {round_dir}"
        )

    def fit(self, parameters, config):
        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = int(config.get("current_round", 0))

        self.logger.info("current_round %s", current_round)

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
            dataset_target_feature=self.dataset_target_feature,
        )

        results = {"client_number": self.client_number}

        data = []

        if current_round <= metadata["num_classes"]:

            if current_round == 1:
                self.net = CNN(cnn_type=self.cnn_type)

                set_weights(self.net, parameters)
                os.makedirs(self.client_model_folder_path, exist_ok=True)
                torch.save(
                    self.net.state_dict(), self.client_model_folder_path + "/model.pth"
                )

                self.net.to("cpu")

            target_class = current_round - 1

            train_dataloader, original_data_length = dataloader.load_dataset_from_disk(
                "train_data",
                self.client_data_folder_path,
                self.batch_size,
                target_class=target_class,
                upsample_amount=1000,
            )

            self.logger.info(
                "Original data length for class %s: %s",
                target_class,
                original_data_length,
            )

            if train_dataloader is None:
                self.logger.info(
                    "No data for target_class %s, skipping VAE training",
                    target_class,
                )
                return (
                    [],
                    0,
                    results,
                )

            if self.mode == "HFedCVAE":
                self.vae = VAE(**metadata["HFedCVAE"]["vae_parameters"])
                results.update(
                    train_vae(
                        vae=self.vae,
                        trainloader=train_dataloader,
                        epochs=self.vae_epochs,
                        device=self.device,
                        dataset_input_feature=self.dataset_input_feature,
                    )
                )

                model = self.vae

            elif self.mode == "HFedCGAN":
                self.generator = Generator(
                    **metadata["HFedCGAN"]["generator_parameters"]
                )
                self.discriminator = Discriminator(
                    **metadata["HFedCGAN"]["discriminator_parameters"]
                )

                results.update(
                    train_gan(
                        generator=self.generator,
                        discriminator=self.discriminator,
                        trainloader=train_dataloader,
                        epochs=self.vae_epochs,
                        device=self.device,
                        dataset_input_feature=self.dataset_input_feature,
                    )
                )

                model = self.generator

            elif self.mode == "HFedCVAEGAN":
                self.vae = VAE(**metadata["HFedCVAE"]["vae_parameters"])
                self.discriminator = Discriminator(
                    **metadata["HFedCGAN"]["discriminator_parameters"]
                )

            data = create_synthetic_data(
                model=model,
                label=target_class,
                device=self.device,
                samples_per_class=original_data_length,
                batch_size=self.batch_size,
                mode=self.mode,
            )

            if self.vae:
                self.vae.to("cpu")
            if self.generator:
                self.generator.to("cpu")
            if self.discriminator:
                self.discriminator.to("cpu")

        elif current_round == metadata["num_classes"] + 1:
            self.net = CNN(cnn_type=self.cnn_type)

            self._set_weights_from_disk("net")

            train_dataloader = dataloader.load_dataset_from_disk(
                "train_data",
                self.client_data_folder_path,
                self.batch_size,
            )

            val_dataloader = dataloader.load_dataset_from_disk(
                "val_data",
                self.client_data_folder_path,
                self.batch_size,
            )

            synthetic_dataloader = dataloader.load_dataset_from_ndarray(
                parameters,
                self.batch_size,
            )

            train_results = train_net(
                net=self.net,
                trainloader=synthetic_dataloader,
                testloader=val_dataloader,
                epochs=self.net_epochs,
                learning_rate=self.net_learning_rate,
                device=self.device,
                dataset_input_feature=self.dataset_input_feature,
                dataset_target_feature=self.dataset_target_feature,
                optimizer_strategy="adam",
            )

            results.update(
                {
                    "synthetic_data_train_loss": train_results["train_loss"],
                    "synthetic_data_train_accuracy": train_results["train_accuracy"],
                }
            )

            data = get_weights(self.net)

            self.net.to("cpu")

        elif current_round > metadata["num_classes"] + 1:
            self.net = CNN(cnn_type=self.cnn_type)

            set_weights(self.net, parameters)

            train_dataloader = dataloader.load_dataset_from_disk(
                "train_data",
                self.client_data_folder_path,
                self.batch_size,
            )

            val_dataloader = dataloader.load_dataset_from_disk(
                "val_data",
                self.client_data_folder_path,
                self.batch_size,
            )

            results.update(
                train_net(
                    net=self.net,
                    trainloader=train_dataloader,
                    testloader=val_dataloader,
                    epochs=self.net_epochs,
                    learning_rate=self.net_learning_rate,
                    device=self.device,
                    dataset_input_feature=self.dataset_input_feature,
                    dataset_target_feature=self.dataset_target_feature,
                    optimizer_strategy="sgd",
                )
            )

            data = get_weights(self.net)

            self.net.to("cpu")

        self.logger.info("results %s", results)

        torch.cuda.empty_cache()

        return (
            data,
            len(train_dataloader.dataset),
            results,
        )

    def evaluate(self, parameters, config):
        self.net = CNN(cnn_type=self.cnn_type)

        current_round = config.get("current_round", -1)

        self.logger.info("current_round %s", current_round)

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
            dataset_target_feature=self.dataset_target_feature,
        )

        test_dataloader = dataloader.load_test_dataset_from_disk(
            self.dataset_folder_path, self.batch_size
        )

        results = {"client_number": self.client_number}

        loss, accuracy = 0, 0

        if current_round <= metadata["num_classes"]:

            synthetic_dataloader = dataloader.load_dataset_from_ndarray(
                parameters,
                self.batch_size,
            )

            self._save_synthetic_images(synthetic_dataloader, current_round)

            train_results = train_net(
                net=self.net,
                trainloader=synthetic_dataloader,
                testloader=test_dataloader,
                epochs=self.net_epochs,
                learning_rate=self.net_learning_rate,
                device=self.device,
                dataset_input_feature=self.dataset_input_feature,
                dataset_target_feature=self.dataset_target_feature,
                optimizer_strategy="adam",
            )

            loss, accuracy = (
                train_results["train_loss"],
                train_results["train_accuracy"],
            )

        elif current_round > metadata["num_classes"]:
            set_weights(self.net, parameters)

            loss, accuracy = test_net(
                self.net,
                test_dataloader,
                self.device,
                self.dataset_input_feature,
                self.dataset_target_feature,
            )

        results.update(
            {
                "loss": loss,
                "accuracy": accuracy,
            }
        )

        self.logger.info("results %s", results)

        self.net.to("cpu")

        torch.cuda.empty_cache()

        return (
            loss,
            len(test_dataloader.dataset),
            results,
        )
