import os

import torch
from datasets import Dataset
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch.utils.data import DataLoader as TorchDataLoader

from src.data_loader import DataLoader
from src.ml_models.cnn import CNN, cnn_config
from src.ml_models.cvae import CVAE
from src.ml_models.train_cvae import train_cvae
from src.ml_models.train_net import test_net, train_net
from src.ml_models.utils import get_device, get_weights, set_weights
from src.scripts.helper import load_metadata, metadata, save_metadata
from src.utils.logger import get_logger


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        client_number,
        batch_size,
        net_epochs,
        cvae_epochs,
        net_learning_rate,
        dataset_folder_path,
        model_folder_path,
        dataset_input_feature,
        dataset_target_feature,
        samples_per_class,
    ):
        super().__init__()
        self.client_number = client_number
        self.batch_size = batch_size
        self.net_epochs = net_epochs
        self.cvae_epochs = cvae_epochs
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
        self.samples_per_class = samples_per_class

        self.cnn_type = list(cnn_config.keys())[
            int(client_number + 1) % len(cnn_config.keys())
        ]
        self.net = CNN(cnn_type=self.cnn_type)
        self.cvae = CVAE()

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

    def _get_dataset_variance(self, dataloader, current_round):
        if current_round == 1:
            dataset = dataloader.dataset
            images = torch.stack(
                [dataset[i][self.dataset_input_feature] for i in range(len(dataset))]
            )
            var = torch.var(images, unbiased=False, dim=[0, 2, 3]).mean().item()

            save_data = {"variance": var}
            save_metadata(save_data, self.client_metadata_path)
        else:
            var = load_metadata(self.client_metadata_path).get("variance", None)

        return var

    def _handle_weights(self):
        cvae_weights = get_weights(self.cvae)
        return cvae_weights

    def _create_synthetic_dataloader(self):
        """
        Memory-efficient synthetic dataloader generator using a CVAE decoder.
        Generates synthetic samples batch-by-batch and streams into a HuggingFace dataset.
        """

        num_classes = metadata["num_classes"]

        self.cvae.eval()
        self.cvae.to(self.device)

        # Total batches per class
        batches_per_class = (
            self.samples_per_class + self.batch_size - 1
        ) // self.batch_size

        def generator():
            """Stream images batch-by-batch to HuggingFace."""
            for cls in range(num_classes):
                for _ in range(batches_per_class):

                    # Current batch size (last batch may be smaller)
                    current_bs = min(self.batch_size, self.samples_per_class)

                    # One-hot condition vector
                    y = torch.zeros(current_bs, num_classes, device=self.device)
                    y[:, cls] = 1

                    # Latent sample
                    z = torch.randn(
                        current_bs, metadata["latent_dim"], device=self.device
                    )
                    with torch.no_grad():
                        z_cond = torch.cat([z, y], dim=1)
                        expanded = self.cvae.fc_expand(z_cond)
                        expanded = expanded.view(-1, *self.cvae.enc_shape)
                        images = self.cvae.decoder(expanded).cpu()

                    # Yield each image + label individually
                    for img in images:
                        yield {
                            self.dataset_input_feature: img,
                            self.dataset_target_feature: cls,
                        }

        # Streaming HF dataset from generator
        hf_dataset = Dataset.from_generator(
            generator,
            features=None,  # Let HF infer tensor types automatically
        )

        # Convert to DataLoader
        loader = TorchDataLoader(
            hf_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        return loader

    def fit(self, parameters, config):
        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = config.get("current_round")
        cvae_index_start = config.get("cvae_index_start")
        cvae_index_end = config.get("cvae_index_end")

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

        set_weights(self.cvae, parameters[cvae_index_start:cvae_index_end])

        results = {"client_number": self.client_number}

        if current_round == 1:
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
                    optimizer_strategy="adam",
                )
            )

            torch.save(
                self.net.state_dict(), self.client_model_folder_path + "/model.pth"
            )

        else:
            x_train_var = self._get_dataset_variance(train_dataloader, current_round)

            results.update(
                train_cvae(
                    cvae=self.cvae,
                    trainloader=train_dataloader,
                    epochs=self.cvae_epochs,
                    device=self.device,
                    dataset_input_feature=self.dataset_input_feature,
                    dataset_target_feature=self.dataset_target_feature,
                    x_train_var=x_train_var,
                )
            )

        weights = self._handle_weights()

        self.logger.info("results %s", results)

        return (
            weights,
            len(train_dataloader.dataset),
            results,
        )

    def evaluate(self, parameters, config):
        current_round = config.get("current_round")
        cvae_index_start = config.get("cvae_index_start")
        cvae_index_end = config.get("cvae_index_end")

        self.logger.info("current_round %s", current_round)

        self._set_weights_from_disk("net")

        set_weights(self.cvae, parameters[cvae_index_start:cvae_index_end])

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
            dataset_target_feature=self.dataset_target_feature,
        )

        test_dataloader = dataloader.load_test_dataset_from_disk(
            self.dataset_folder_path, self.batch_size
        )

        results = {"client_number": self.client_number}

        loss, acc = 0.0, 0.0

        if current_round != 1:
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

            synthetic_dataloader = self._create_synthetic_dataloader()

            train_results = train_net(
                net=self.net,
                trainloader=synthetic_dataloader,
                testloader=val_dataloader,
                epochs=self.net_epochs,
                learning_rate=self.net_learning_rate,
                device=self.device,
                dataset_input_feature=self.dataset_input_feature,
                dataset_target_feature=self.dataset_target_feature,
                optimizer_strategy="sgd",
            )

            results.update(
                {
                    "synthetic_train_loss": train_results["train_loss"],
                    "synthetic_train_accuracy": train_results["train_accuracy"],
                }
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


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    client_number = context.node_config.get("partition-id")
    batch_size = context.run_config.get("batch-size")
    net_epochs = context.run_config.get("net-epochs")
    cvae_epochs = context.run_config.get("cvae-epochs")
    net_learning_rate = context.run_config.get("net-learning-rate")
    dataset_folder_path = context.run_config.get("dataset-folder-path")
    model_folder_path = context.run_config.get("model-folder-path")
    dataset_input_feature = context.run_config.get("dataset-input-feature")
    dataset_target_feature = context.run_config.get("dataset-target-feature")
    samples_per_class = context.run_config.get("samples-per-class")

    # Return Client instance
    return FlowerClient(
        client_number,
        batch_size,
        net_epochs,
        cvae_epochs,
        net_learning_rate,
        dataset_folder_path,
        model_folder_path,
        dataset_input_feature,
        dataset_target_feature,
        samples_per_class,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
