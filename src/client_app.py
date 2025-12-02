import os

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.data_loader import DataLoader
from src.ml_models.cnn import CNN, cnn_config
from src.ml_models.gated_pixelcnn import GatedPixelCNN
from src.ml_models.train_vqvae_with_pixel_cnn import train_vqvae_with_pixel_cnn
from src.ml_models.utils import (
    generate_and_save_images,
    get_device,
    get_weights,
    set_weights,
)
from src.ml_models.vqvae import VQVAE
from src.scripts.helper import load_metadata, save_metadata
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
        self.client_metadata_path = os.path.join(
            self.client_data_folder_path, "metadata.json"
        )
        self.dataset_input_feature = dataset_input_feature
        self.dataset_target_feature = dataset_target_feature

        self.cnn_type = list(cnn_config.keys())[
            int(client_number + 1) % len(cnn_config.keys())
        ]
        self.net = CNN(cnn_type=self.cnn_type)
        self.vqvae = VQVAE()
        self.pixel_cnn = GatedPixelCNN()

        self.device = get_device()

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

    def _get_dataset_variance(self, dataloader, current_round):
        if current_round == 1:
            dataset = dataloader.dataset
            images = torch.stack(
                [dataset[i][self.dataset_input_feature] for i in range(len(dataset))]
            )
            var, _ = torch.var_mean(images)
            var = var.item()

            save_data = {"variance": var}
            save_metadata(save_data, self.client_metadata_path)
        else:
            var = load_metadata(self.client_metadata_path).get("variance", None)

        return var

    def _handle_weights(self):
        vqvae_weights = get_weights(self.vqvae)
        pixel_cnn_weights = get_weights(self.pixel_cnn)

        return vqvae_weights + pixel_cnn_weights

    def fit(self, parameters, config):
        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = config.get("current_round")
        vqvae_index_start = config.get("vqvae_index_start")
        vqvae_index_end = config.get("vqvae_index_end")
        pixel_cnn_index_start = config.get("pixel_cnn_index_start")
        pixel_cnn_index_end = config.get("pixel_cnn_index_end")

        self.logger.info("current_round %s", current_round)

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
            dataset_target_feature=self.dataset_target_feature,
        )

        train_dataloader = dataloader.load_dataset_from_disk(
            "train_data",
            self.client_data_folder_path,
            self.batch_size,
        )
        x_train_var = self._get_dataset_variance(train_dataloader, current_round)

        set_weights(self.vqvae, parameters[vqvae_index_start:vqvae_index_end])
        set_weights(
            self.pixel_cnn,
            parameters[pixel_cnn_index_start:pixel_cnn_index_end],
        )

        train_vqvae_with_pixel_cnn_results = train_vqvae_with_pixel_cnn(
            vqvae=self.vqvae,
            pixel_cnn=self.pixel_cnn,
            trainloader=train_dataloader,
            epochs=self.local_epochs,
            device=self.device,
            dataset_input_feature=self.dataset_input_feature,
            dataset_target_feature=self.dataset_target_feature,
            x_train_var=x_train_var,
        )

        train_vqvae_with_pixel_cnn_results.update({"client_number": self.client_number})

        self.logger.info(
            "train_vqvae_with_pixel_cnn_results %s", train_vqvae_with_pixel_cnn_results
        )

        weights = self._handle_weights()

        return (
            weights,
            len(train_dataloader.dataset),
            train_vqvae_with_pixel_cnn_results,
        )

    def evaluate(self, parameters, config):
        current_round = config.get("current_round")
        vqvae_index_start = config.get("vqvae_index_start")
        vqvae_index_end = config.get("vqvae_index_end")
        pixel_cnn_index_start = config.get("pixel_cnn_index_start")
        pixel_cnn_index_end = config.get("pixel_cnn_index_end")

        self.logger.info("current_round %s", current_round)

        # if current_round == 1:
        #     os.makedirs(self.client_model_folder_path, exist_ok=True)
        # else:
        #     self._set_weights_from_disk()

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
        )

        test_dataloader = dataloader.load_test_dataset_from_disk(
            self.dataset_folder_path, self.batch_size
        )

        client_metadata = load_metadata(self.client_metadata_path)

        set_weights(self.vqvae, parameters[vqvae_index_start:vqvae_index_end])
        set_weights(
            self.pixel_cnn,
            parameters[pixel_cnn_index_start:pixel_cnn_index_end],
        )

        # Generate and save images for all 10 classes (10 images per class)
        synthetic_images_folder = os.path.join(
            self.client_model_folder_path, "synthetic_images"
        )
        generate_and_save_images(
            vqvae=self.vqvae,
            pixel_cnn=self.pixel_cnn,
            device=self.device,
            output_folder=synthetic_images_folder,
            current_round=current_round,
            num_classes=10,
            images_per_class=10,
        )

        self.logger.info(
            "Generated and saved synthetic images for round %s", current_round
        )

        evaluate_results = {}

        evaluate_results.update({"client_number": self.client_number})
        self.logger.info("evaluate_results %s", evaluate_results)

        return (
            evaluate_results.get("classification_loss", 0.0),
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
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
