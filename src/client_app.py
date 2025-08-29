import os

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from PIL import Image

from src.data_loader import DataLoader
from src.ml_models.decoder import Decoder, DecoderLatentSpace
from src.ml_models.HFedCVAE import HFedCVAE, cnn_config
from src.ml_models.train_test_classification import train_classification
from src.ml_models.train_test_conditional_gan_with_classification import (
    test_conditional_gan_with_classification,
    train_conditional_gan_with_classification,
)
from src.ml_models.train_test_conditional_vae_with_classification import (
    test_conditional_vae_with_classification,
    train_conditional_vae_with_classification,
)
from src.ml_models.utils import to_onehot
from src.scripts.helper import metadata
from src.utils.logger import get_logger


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        client_number,
        batch_size,
        local_epochs,
        learning_rate,
        dataset_folder_path,
        model_folder_path,
        dataset_input_feature,
        dataset_target_feature,
        max_synthetic_data_per_client,
        generator_mode,
    ):
        super().__init__()
        self.client_number = client_number
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.dataset_folder_path = dataset_folder_path
        self.client_data_folder_path = os.path.join(
            dataset_folder_path, f"client_{client_number}"
        )
        self.client_model_folder_path = os.path.join(
            model_folder_path, f"client_{client_number}"
        )
        self.dataset_input_feature = dataset_input_feature
        self.dataset_target_feature = dataset_target_feature
        self.max_synthetic_data_per_client = max_synthetic_data_per_client
        self.generator_mode = generator_mode

        self.cnn_type = list(cnn_config.keys())[
            int(client_number + 1) % len(cnn_config.keys())
        ]
        self.net = HFedCVAE(cnn_type=self.cnn_type)
        if self.generator_mode == "vae":
            self.decoderLatentSpace = DecoderLatentSpace(
                in_latent_dim=cnn_config[self.cnn_type]["fc2_out_features"]
            )
        self.decoder = Decoder()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Configure logging
        self.logger = get_logger(f"{__name__}_Client_{client_number}", client_number)
        self.logger.info("Client %s initiated", self.client_number)

    def _set_weights_from_disk(self):
        self.net.load_state_dict(
            torch.load(
                self.client_model_folder_path + "/model.pth",
                map_location="cpu",
            )
        )
        if self.generator_mode == "vae":
            self.decoderLatentSpace.load_state_dict(
                torch.load(
                    self.client_model_folder_path + "/decoder_latent_space.pth",
                    map_location="cpu",
                )
            )
        self.decoder.load_state_dict(
            torch.load(
                self.client_model_folder_path + "/decoder.pth",
                map_location="cpu",
            )
        )

    def _generate_synthetic_dataset(self, train_dataloader, current_round):

        synthetic_data_len = min(
            self.max_synthetic_data_per_client, len(train_dataloader.dataset)
        )

        self.logger.info("synthetic_data_len %s", synthetic_data_len)

        labels_np = np.array(train_dataloader.dataset[self.dataset_target_feature])
        labels, counts = np.unique(labels_np, return_counts=True)
        label_probs = counts / counts.sum()

        self.logger.info(
            "labels-%s, counts-%s, label_probs-%s", labels, counts, label_probs
        )

        # Create base directory for saving synthetic images
        synthetic_images_base_dir = os.path.join(
            self.client_model_folder_path, "synthetic_images"
        )
        os.makedirs(synthetic_images_base_dir, exist_ok=True)

        self.net.eval()
        num_samples_per_class = np.round(label_probs * synthetic_data_len).astype(int)
        self.logger.info(
            "Generating synthetic data with counts per class: %s", num_samples_per_class
        )

        synthetic_images = []
        synthetic_labels = []
        image_counter = 0

        with torch.no_grad():
            for index, count in enumerate(num_samples_per_class):
                if count == 0:
                    continue

                remaining = count
                valid_samples_collected = 0

                while remaining > 0:
                    batch_count = min(128, remaining)
                    y = torch.full(
                        (batch_count,),
                        labels[index],
                        dtype=torch.long,
                        device=self.device,
                    )
                    y_onehot = to_onehot(y, metadata["num_classes"], device=self.device)
                    z = torch.randn(
                        batch_count, metadata["decoder_latent_dim"], device=self.device
                    )

                    # Generate synthetic samples on GPU
                    samples = self.decoder(z, y_onehot)  # (B, 1, 28, 28)

                    # Filter samples by checking model predictions
                    logits, _ = self.net(samples)
                    predicted_labels = torch.argmax(logits, dim=1)

                    # Keep only samples where model predicts the correct label
                    correct_predictions = predicted_labels == y

                    for i in range(batch_count):
                        if correct_predictions[i]:
                            # Convert tensor to numpy and save as PNG
                            image_data = samples[i].cpu().numpy().round(4)
                            synthetic_images.append(image_data)
                            synthetic_labels.append(labels[index])

                            # Save as PNG file in round/class/image structure
                            # Assume image is single channel (grayscale) with shape (1, H, W)
                            if len(image_data.shape) == 3 and image_data.shape[0] == 1:
                                # Create round and class directories
                                round_dir = os.path.join(
                                    synthetic_images_base_dir, f"round_{current_round}"
                                )
                                class_dir = os.path.join(
                                    round_dir, f"class_{labels[index]}"
                                )
                                os.makedirs(class_dir, exist_ok=True)

                                # Remove channel dimension and convert to 0-255 range
                                image_2d = image_data.squeeze(0)  # (H, W)
                                # Normalize to 0-255 range (assuming input is in 0-1 range)
                                image_2d = np.clip(image_2d * 255, 0, 255).astype(
                                    np.uint8
                                )

                                # Create PIL Image and save
                                pil_image = Image.fromarray(image_2d, mode="L")
                                filename = f"client_{self.client_number}_class_{labels[index]}_sample_{image_counter:06d}.png"
                                filepath = os.path.join(class_dir, filename)
                                pil_image.save(filepath)

                                image_counter += 1

                            valid_samples_collected += 1

                    remaining -= batch_count

                self.logger.info(
                    "Class %s: Generated %d valid samples out of %d target samples",
                    labels[index],
                    valid_samples_collected,
                    count,
                )

        # Convert to numpy arrays - these are homogeneous arrays that can be serialized
        synthetic_images_array = np.array(synthetic_images)
        synthetic_labels_array = np.array(synthetic_labels)

        self.logger.info(
            "Total filtered synthetic data: %d images, %d labels",
            len(synthetic_images_array),
            len(synthetic_labels_array),
        )

        self.logger.info(
            "Saved %d synthetic images as PNG files in %s",
            image_counter,
            synthetic_images_base_dir,
        )

        return (synthetic_images_array, synthetic_labels_array)

    def fit(self, parameters, config):
        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = config.get("current_round")
        self.logger.info("current_round %s", current_round)

        if current_round == 1:
            os.makedirs(self.client_model_folder_path, exist_ok=True)
        else:
            self._set_weights_from_disk()

        # Create data loader instance for transforms
        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
            dataset_target_feature=self.dataset_target_feature,
        )

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

        if self.generator_mode == "gan":
            train_results = train_conditional_gan_with_classification(
                net=self.net,
                trainloader=train_dataloader,
                testloader=val_dataloader,
                epochs=self.local_epochs,
                learning_rate=self.learning_rate,
                device=self.device,
                dataset_input_feature=self.dataset_input_feature,
                dataset_target_feature=self.dataset_target_feature,
                decoder=self.decoder,
            )
        elif self.generator_mode == "vae":
            train_results = train_conditional_vae_with_classification(
                net=self.net,
                trainloader=train_dataloader,
                testloader=val_dataloader,
                epochs=self.local_epochs,
                learning_rate=self.learning_rate,
                device=self.device,
                dataset_input_feature=self.dataset_input_feature,
                dataset_target_feature=self.dataset_target_feature,
                decoderLatentSpace=self.decoderLatentSpace,
                decoder=self.decoder,
            )
        if train_results is None:
            train_results = {}

        train_results.update({"client_number": self.client_number})

        self.logger.info("train_results %s", train_results)

        synthetic_data = self._generate_synthetic_dataset(
            train_dataloader, current_round
        )

        if current_round != 1:
            self.logger.info(
                "Received synthetic data: images shape %s, labels shape %s",
                parameters[0].shape,
                parameters[1].shape,
            )

            synthetic_dataloader = dataloader.load_dataset_from_ndarray(
                parameters, self.batch_size
            )

            self.logger.info(
                "Created synthetic DataLoader with %d samples",
                len(synthetic_dataloader.dataset),
            )

            # Log data distribution of synthetic dataloader dataset
            synthetic_labels = np.array(
                synthetic_dataloader.dataset[self.dataset_target_feature]
            )
            synthetic_unique_labels, synthetic_counts = np.unique(
                synthetic_labels, return_counts=True
            )
            synthetic_label_probs = synthetic_counts / synthetic_counts.sum()

            self.logger.info(
                "Synthetic dataset distribution - labels: %s, counts: %s, probabilities: %s",
                synthetic_unique_labels,
                synthetic_counts,
                synthetic_label_probs,
            )

            train_results_on_synthetic_data = train_classification(
                net=self.net,
                trainloader=synthetic_dataloader,
                testloader=val_dataloader,
                epochs=self.local_epochs,
                learning_rate=self.learning_rate,
                device=self.device,
                dataset_input_feature=self.dataset_input_feature,
                dataset_target_feature=self.dataset_target_feature,
            )

            self.logger.info(
                "train_results_on_synthetic_data %s", train_results_on_synthetic_data
            )

        torch.save(self.net.state_dict(), self.client_model_folder_path + "/model.pth")
        if self.generator_mode == "vae":
            torch.save(
                self.decoderLatentSpace.state_dict(),
                self.client_model_folder_path + "/decoder_latent_space.pth",
            )
        torch.save(
            self.decoder.state_dict(), self.client_model_folder_path + "/decoder.pth"
        )

        return (
            [synthetic_data[0], synthetic_data[1]],
            len(train_dataloader.dataset),
            train_results,
        )

    def evaluate(self, parameters, config):
        current_round = config.get("current_round")

        self.logger.info("current_round %s", current_round)

        self._set_weights_from_disk()

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
        )

        test_dataloader = dataloader.load_test_dataset_from_disk(
            self.dataset_folder_path, self.batch_size
        )

        if self.generator_mode == "gan":
            evaluate_results = test_conditional_gan_with_classification(
                net=self.net,
                decoder=self.decoder,
                testloader=test_dataloader,
                device=self.device,
                dataset_input_feature=self.dataset_input_feature,
                dataset_target_feature=self.dataset_target_feature,
            )
        elif self.generator_mode == "vae":
            evaluate_results = test_conditional_vae_with_classification(
                net=self.net,
                decoderLatentSpace=self.decoderLatentSpace,
                decoder=self.decoder,
                testloader=test_dataloader,
                device=self.device,
                dataset_input_feature=self.dataset_input_feature,
                dataset_target_feature=self.dataset_target_feature,
            )

        if evaluate_results is None:
            evaluate_results = {}

        evaluate_results.update({"client_number": self.client_number})
        self.logger.info("evaluate_results %s", evaluate_results)

        return (
            evaluate_results["classification_loss"],
            len(test_dataloader.dataset),
            evaluate_results,
        )


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    client_number = context.node_config.get("partition-id")
    batch_size = context.run_config.get("batch-size")
    local_epochs = context.run_config.get("local-epochs")
    learning_rate = context.run_config.get("learning-rate")
    dataset_folder_path = context.run_config.get("dataset-folder-path")
    model_folder_path = context.run_config.get("model-folder-path")
    dataset_input_feature = context.run_config.get("dataset-input-feature")
    dataset_target_feature = context.run_config.get("dataset-target-feature")
    max_synthetic_data_per_client = context.run_config.get(
        "max-synthetic-data-per-client"
    )
    generator_mode = context.run_config.get("generator-mode")

    # Return Client instance
    return FlowerClient(
        client_number,
        batch_size,
        local_epochs,
        learning_rate,
        dataset_folder_path,
        model_folder_path,
        dataset_input_feature,
        dataset_target_feature,
        max_synthetic_data_per_client,
        generator_mode,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
