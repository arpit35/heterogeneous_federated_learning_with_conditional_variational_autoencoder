import os

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.data_loader import DataLoader
from src.ml_models.cnn import cnn_config
from src.ml_models.HFedCVAE import HFedCVAE
from src.ml_models.utils import get_weights, set_weights, test, train
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
        self.cnn_type = list(cnn_config.keys())[
            int(client_number + 1) % len(cnn_config.keys())
        ]
        self.net = HFedCVAE(cnn_type=self.cnn_type)
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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Configure logging
        self.logger = get_logger(f"{__name__}_Client_{client_number}", client_number)
        self.logger.info("Client %s initiated", self.client_number)

    def _set_weights_from_disk(self, current_round):
        if current_round != 1:
            self.net.load_state_dict(
                torch.load(
                    self.client_model_folder_path + "/model.pth",
                    map_location="cpu",
                )
            )

    def fit(self, parameters, config):
        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = config.get("current_round")

        self._set_weights_from_disk(current_round)

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
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

        train_results = train(
            self.net,
            train_dataloader,
            val_dataloader,
            self.local_epochs,
            self.learning_rate,
            self.device,
            self.dataset_input_feature,
            self.dataset_target_feature,
        )

        torch.save(self.net.state_dict(), self.client_model_folder_path + "/model.pth")

        return (
            all_weights,
            len(train_dataloader.dataset),
            train_results,
        )

    def evaluate(self, parameters, config):
        self.net.load_state_dict(
            torch.load(self.client_model_folder_path + "/model.pth", map_location="cpu")
        )
        current_round = config.get("current_round")

        self._set_weights_from_disk(current_round)

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
        )

        test_dataloader = dataloader.load_test_dataset_from_disk(
            self.dataset_folder_path, self.batch_size
        )

        loss, accuracy = test(
            self.net,
            test_dataloader,
            self.device,
            self.dataset_input_feature,
            self.dataset_target_feature,
        )

        return (
            loss,
            len(test_dataloader.dataset),
            {
                "accuracy": accuracy,
                "client_number": self.client_number,
            },
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
