import os

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.data_loader import DataLoader
from src.ml_models.HFedCVAE import HFedCVAE, cnn_config
from src.ml_models.utils import test, to_onehot, train
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
        self.max_synthetic_data_per_client = max_synthetic_data_per_client

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Configure logging
        self.logger = get_logger(f"{__name__}_Client_{client_number}", client_number)
        self.logger.info("Client %s initiated", self.client_number)

    def _set_weights_from_disk(self, current_round):
        if current_round == 1:
            os.makedirs(self.client_model_folder_path, exist_ok=True)
        else:
            self.net.load_state_dict(
                torch.load(
                    self.client_model_folder_path + "/model.pth",
                    map_location="cpu",
                )
            )

    def _generate_synthetic_dataset(self, train_dataloader):
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

        self.net.eval()
        num_samples_per_class = np.round(label_probs * synthetic_data_len).astype(int)
        self.logger.info(
            "Generating synthetic data with counts per class: %s", num_samples_per_class
        )

        synthetic_images = []
        synthetic_labels = []

        with torch.no_grad():
            for index, count in enumerate(num_samples_per_class):
                if count == 0:
                    continue

                remaining = count
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
                    samples = self.net.decoder(z, y_onehot)  # (B, 1, 28, 28)

                    for i in range(batch_count):
                        synthetic_images.append(
                            samples[i].cpu().numpy().astype(np.uint8)
                        )
                        synthetic_labels.append(labels[index])

                    remaining -= batch_count

        # Convert to numpy arrays - these are homogeneous arrays that can be serialized
        synthetic_images_array = np.array(synthetic_images)
        synthetic_labels_array = np.array(synthetic_labels)

        return (synthetic_images_array, synthetic_labels_array)

    def fit(self, parameters, config):
        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = config.get("current_round")
        self.logger.info("current_round %s", current_round)

        self._set_weights_from_disk(current_round)

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

            train_results_on_synthetic_data = train(
                self.net,
                synthetic_dataloader,
                val_dataloader,
                self.local_epochs,
                self.learning_rate,
                self.device,
                self.dataset_input_feature,
                self.dataset_target_feature,
                detach_decoder=True,
            )

            self.logger.info(
                "train_results_on_synthetic_data %s", train_results_on_synthetic_data
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

        self.logger.info("train_results %s", train_results)

        torch.save(self.net.state_dict(), self.client_model_folder_path + "/model.pth")

        synthetic_data = self._generate_synthetic_dataset(train_dataloader)

        return (
            [synthetic_data[0], synthetic_data[1]],
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

        combined_loss, accuracy, classification_loss, BCE, KLD = test(
            self.net,
            test_dataloader,
            self.device,
            self.dataset_input_feature,
            self.dataset_target_feature,
        )

        evaluate_results = {
            "client_number": self.client_number,
            "combined_loss": combined_loss,
            "accuracy": accuracy,
            "classification_loss": classification_loss,
            "BCE": BCE,
            "KLD": KLD,
        }

        self.logger.info("evaluate_results %s", evaluate_results)

        return (
            classification_loss,
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
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
