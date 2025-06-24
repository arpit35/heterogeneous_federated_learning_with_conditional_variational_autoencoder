import os

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.data_loader import DataLoader
from src.ml_models.cnn import cnn_config
from src.ml_models.HFedPVA import HFedPVA
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
        self.net = HFedPVA(cnn_type=self.cnn_type)
        self.client_number = client_number
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
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

    def fit(self, parameters, config):
        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = config.get("current_round")

        if current_round == 1:
            os.makedirs(self.client_model_folder_path, exist_ok=True)

            generic_encoder_start = config.get("generic_encoder_start")
            generic_encoder_end = config.get("generic_encoder_end")
            personalized_encoder_start = config.get("personalized_encoder_start")
            personalized_encoder_end = config.get("personalized_encoder_end")
            decoder_start = config.get("decoder_start")
            decoder_end = config.get("decoder_end")

            set_weights(
                self.net.generic_encoder,
                parameters[generic_encoder_start:generic_encoder_end],
            )

            set_weights(
                self.net.personalized_encoder,
                parameters[personalized_encoder_start:personalized_encoder_end],
            )

            set_weights(
                self.net.decoder,
                parameters[decoder_start:decoder_end],
            )

        else:
            self.net.load_state_dict(
                torch.load(
                    self.client_model_folder_path + "/model.pth",
                    map_location="cpu",
                )
            )
            set_weights(self.net.personalized_encoder, parameters)

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
        )

        train_dataloader = dataloader.load_dataset_from_disk(
            "train_data",
            self.client_data_folder_path,
            self.batch_size,
        )
        test_dataloader = dataloader.load_dataset_from_disk(
            "test_data",
            self.client_data_folder_path,
            self.batch_size,
        )

        train_results = train(
            self.net,
            train_dataloader,
            test_dataloader,
            self.local_epochs,
            self.learning_rate,
            self.device,
            self.dataset_input_feature,
            self.dataset_target_feature,
        )

        torch.save(self.net.state_dict(), self.client_model_folder_path + "/model.pth")

        return (
            get_weights(self.net.personalized_encoder),
            len(train_dataloader.dataset),
            train_results,
        )

    def evaluate(self, parameters, config):
        self.net.load_state_dict(
            torch.load(self.client_model_folder_path + "/model.pth", map_location="cpu")
        )
        set_weights(self.net.personalized_encoder, parameters)

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
        )

        test_dataloader = dataloader.load_dataset_from_disk(
            "test_data",
            self.client_data_folder_path,
            self.batch_size,
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
