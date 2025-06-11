import os

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.data_loader import DataLoader
from src.ml_models.cnn import CNN, cnn_config
from src.ml_models.utils import get_weights, set_weights
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
        dataset_input_feature,
        dataset_target_feature,
    ):
        super().__init__()
        self.net = CNN(
            cnn_type=list(cnn_config.keys())[
                int(client_number) % len(cnn_config.keys())
            ]
        )
        self.client_number = client_number
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.client_folder_path = os.path.join(
            dataset_folder_path, f"client_{client_number}"
        )
        self.dataset_input_feature = dataset_input_feature
        self.dataset_target_feature = dataset_target_feature

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Configure logging
        self.logger = get_logger(f"{__name__}_Client_{client_number}", client_number)
        self.logger.info("Client %s initiated", self.client_number)

    def fit(self, parameters, config):
        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = config.get("current_round", 0)

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
            dataset_num_channels=self.dataset_num_channels,
            model_input_image_size=self.model_input_image_size,
        )

        train_dataloader = dataloader.load_dataset_from_disk(
            "train_data",
            self.client_folder_path,
            self.num_batches_each_round,
            self.batch_size,
            self.gradient_accumulation_steps,
        )
        val_dataloader = dataloader.load_dataset_from_disk(
            "val_data",
            self.client_folder_path,
            self.num_batches_each_round,
            self.batch_size,
            self.gradient_accumulation_steps,
        )

        set_weights(self.net, parameters)

        train_results = train(
            self.net,
            train_dataloader,
            val_dataloader,
            self.local_epochs,
            learning_rate,
            self.device,
            momentum,
            self.dataset_input_feature,
            self.dataset_target_feature,
            self.gradient_accumulation_steps,
            sga,
        )

        return (
            get_weights(self.net),
            len(train_dataloader.dataset),
            train_results,
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
            dataset_num_channels=self.dataset_num_channels,
            model_input_image_size=self.model_input_image_size,
        )

        val_dataloader = dataloader.load_dataset_from_disk(
            file_name,
            self.client_folder_path,
            self.num_batches_each_round,
            self.batch_size,
            self.gradient_accumulation_steps,
        )

        loss, accuracy = test(
            self.net,
            val_dataloader,
            self.device,
            self.dataset_input_feature,
            self.dataset_target_feature,
        )

        return (
            loss,
            len(val_dataloader.dataset),
            {
                "accuracy": accuracy,
                "client_number": self.client_number,
            },
        )


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    client_number = context.node_config.get("partition-id", None)
    batch_size = context.run_config.get("batch-size", None)
    local_epochs = context.run_config.get("local-epochs", None)
    learning_rate = context.run_config.get("learning-rate", None)
    dataset_folder_path = context.run_config.get("dataset-folder-path", None)
    dataset_input_feature = context.run_config.get("dataset-input-feature", None)
    dataset_target_feature = context.run_config.get("dataset-target-feature", None)

    # Return Client instance
    return FlowerClient(
        client_number,
        batch_size,
        local_epochs,
        learning_rate,
        dataset_folder_path,
        dataset_input_feature,
        dataset_target_feature,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
