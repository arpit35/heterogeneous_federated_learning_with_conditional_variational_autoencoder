import os

import torch
from flwr.client import NumPyClient

from src.data_loader import DataLoader
from src.ml_models.cnn import CNN, cnn_config
from src.ml_models.h_fed_pfs import FPN, Adapter
from src.ml_models.train_h_fed_pfs import train_h_fed_pfs
from src.ml_models.train_net import test_net
from src.ml_models.utils import get_device, get_weights, set_weights
from src.utils.logger import get_logger


class FlowerHFedPFSClient(NumPyClient):
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

        self.cnn_type = list(cnn_config.keys())[
            int(client_number + 1) % len(cnn_config.keys())
        ]
        self.net = CNN(cnn_type=self.cnn_type)
        self.fpn = FPN()
        self.adapter = Adapter()

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
        if model == "fpn":
            self.fpn.load_state_dict(
                torch.load(
                    self.client_model_folder_path + "/fpn.pth",
                    map_location="cpu",
                )
            )

    def fit(self, parameters, config):
        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = int(config.get("current_round", 0))

        self.logger.info("current_round %s", current_round)

        set_weights(self.adapter, parameters)

        if current_round != 1:
            self._set_weights_from_disk("net")
            self._set_weights_from_disk("fpn")

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
            train_h_fed_pfs(
                net=self.net,
                fpn=self.fpn,
                adapter=self.adapter,
                trainloader=train_dataloader,
                testloader=val_dataloader,
                epochs=self.net_epochs,
                learning_rate=self.net_learning_rate,
                device=self.device,
                dataset_input_feature=self.dataset_input_feature,
                dataset_target_feature=self.dataset_target_feature,
                optimizer_strategy="adam",
            )
        )
        # Save the trained model to disk
        os.makedirs(self.client_model_folder_path, exist_ok=True)
        torch.save(self.net.state_dict(), self.client_model_folder_path + "/model.pth")
        torch.save(self.fpn.state_dict(), self.client_model_folder_path + "/fpn.pth")

        self.logger.info("results %s", results)

        return (
            get_weights(self.adapter),
            len(train_dataloader.dataset),
            results,
        )

    def evaluate(self, parameters, config):
        current_round = config.get("current_round", -1)

        self.logger.info("current_round %s", current_round)

        self._set_weights_from_disk("net")

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

        return (
            loss,
            len(test_dataloader.dataset),
            results,
        )
