import os

import torch
from flwr.client import NumPyClient

from src.data_loader import DataLoader
from src.ml_models.cnn import CNN
from src.ml_models.train_net import test_net, train_net
from src.ml_models.utils import get_device, get_weights, set_weights
from src.utils.logger import get_logger


class FlowerDefaultClient(NumPyClient):
    def __init__(
        self,
        client_number,
        batch_size,
        net_epochs,
        net_learning_rate,
        dataset_folder_path,
        model_folder_path,
        dataset_input_feature,
        dataset_target_feature,
        cnn_type,
    ):
        super().__init__()
        self.client_number = client_number
        self.batch_size = batch_size
        self.net_epochs = net_epochs
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

        self.net = CNN(cnn_type=self.cnn_type)

        self.device = get_device()

        # Configure logging
        self.logger = get_logger(f"{__name__}_Client_{client_number}", client_number)
        self.logger.info("Client %s initiated", self.client_number)

    def fit(self, parameters, config):
        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = int(config.get("current_round", 0))

        self.logger.info("current_round %s", current_round)

        set_weights(self.net, parameters)

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

        results = {"client_number": self.client_number}

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

        self.logger.info("results %s", results)

        self.net.to("cpu")

        torch.cuda.empty_cache()

        return (
            get_weights(self.net),
            len(train_dataloader.dataset),
            results,
        )

    def evaluate(self, parameters, config):
        current_round = config.get("current_round", -1)

        self.logger.info("current_round %s", current_round)

        set_weights(self.net, parameters)

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
            dataset_target_feature=self.dataset_target_feature,
        )

        test_dataloader = dataloader.load_test_dataset_from_disk(
            self.dataset_folder_path, self.batch_size
        )

        results = {"client_number": self.client_number}

        loss, acc = test_net(
            net=self.net,
            testloader=test_dataloader,
            device=self.device,
            dataset_input_feature=self.dataset_input_feature,
            dataset_target_feature=self.dataset_target_feature,
        )

        results.update({"loss": loss, "accuracy": acc})

        self.logger.info("results %s", results)

        self.net.to("cpu")

        torch.cuda.empty_cache()

        return (
            loss,
            len(test_dataloader.dataset),
            results,
        )
