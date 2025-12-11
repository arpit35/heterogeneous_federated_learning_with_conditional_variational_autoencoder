import os

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays, ndarrays_to_parameters
from PIL import Image
from torch.utils.data import DataLoader as TorchDataLoader

from src.data_loader import DataLoader
from src.ml_models.cnn import CNN, cnn_config
from src.ml_models.train_net import test_net, train_net
from src.ml_models.train_vae import train_vae
from src.ml_models.utils import get_device, set_weights
from src.ml_models.vae import vae
from src.scripts.helper import metadata
from src.utils.logger import get_logger


# Define Flower Client
class FlowerClient(NumPyClient):
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
        samples_per_class,
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
        self.samples_per_class = samples_per_class

        self.cnn_type = list(cnn_config.keys())[
            int(client_number + 1) % len(cnn_config.keys())
        ]
        self.net = CNN(cnn_type=self.cnn_type)
        self.vae = vae()

        self.device = get_device()

        self.net_training_starting_round = 1
        self.vae_training_starting_round = 2

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
        import os

        import torch
        from PIL import Image

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

    def _create_synthetic_data(self, label, return_dataloader=False) -> NDArrays | TorchDataLoader:
        num_classes = metadata["num_classes"]

        self.vae.eval()
        self.vae.to(self.device)

        total_samples = self.samples_per_class * num_classes

        # Create synthetic data in memory (or in batches if memory is constrained)
        synthetic_data = []
        synthetic_labels = []

        # Generate in batches to be memory efficient
        for start_idx in range(0, total_samples, self.batch_size):
            current_batch_size = min(self.batch_size, total_samples - start_idx)

            # Sample z
            z = torch.randn(
                current_batch_size, metadata["latent_dim"], device=self.device
            )

            # Decode
            with torch.no_grad():
                expanded = self.vae.fc_expand(z)
                expanded = expanded.view(current_batch_size, *self.vae.enc_shape)
                images = self.vae.decoder(expanded)

            synthetic_data.append(images.cpu())
            synthetic_labels.append(
                torch.tensor(
                    [label for _ in range(current_batch_size)],
                    dtype=torch.long,
                ).cpu()
            )

        # Concatenate all batches
        synthetic_data = torch.cat(synthetic_data, dim=0)
        synthetic_labels = torch.cat(synthetic_labels, dim=0)

        if not return_dataloader:
            return [synthetic_data.numpy(), synthetic_labels.numpy()]

        # Create a proper PyTorch Dataset
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, data, labels, input_feature, target_feature):
                self.data = data
                self.labels = labels
                self.input_feature_name = input_feature
                self.target_feature_name = target_feature

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {
                    self.input_feature_name: self.data[idx],
                    self.target_feature_name: self.labels[idx],
                }

        # Create dataset and dataloader
        dataset = SyntheticDataset(
            synthetic_data,
            synthetic_labels,
            self.dataset_input_feature,
            self.dataset_target_feature,
        )
        dataloader = TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        return dataloader

    def fit(self, parameters, config):
        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = int(config.get("current_round", 0))

        self.logger.info("current_round %s", current_round)

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
            dataset_target_feature=self.dataset_target_feature,
        )

        results = {"client_number": self.client_number}

        if current_round >= self.net_training_starting_round:
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
                    optimizer_strategy="adam",
                )
            )
            # Save the trained model to disk
            os.makedirs(self.client_model_folder_path, exist_ok=True)
            torch.save(
                self.net.state_dict(), self.client_model_folder_path + "/model.pth"
            )

        elif current_round >= self.vae_training_starting_round:

            target_class = current_round - self.vae_training_starting_round

            train_dataloader = dataloader.load_dataset_from_disk(
                "train_data",
                self.client_data_folder_path,
                self.batch_size,
                target_class=target_class,
                upsample_amount=1000,
            )

            results.update(
                train_vae(
                    vae=self.vae,
                    trainloader=train_dataloader,
                    epochs=self.vae_epochs,
                    device=self.device,
                    dataset_input_feature=self.dataset_input_feature,
                    dataset_target_feature=self.dataset_target_feature,
                )
            )

            synthetic_data = self._create_synthetic_data(
                label=target_class, return_dataloader=False
            )

            synthetic_data = ndarrays_to_parameters(synthetic_data)

        self.logger.info("results %s", results)

        return (
            synthetic_data,
            len(train_dataloader.dataset),
            results,
        )

    def evaluate(self, parameters, config):
        current_round = config.get("current_round", -1)
        vae_index_start = config.get("vae_index_start")
        vae_index_end = config.get("vae_index_end")

        self.logger.info("current_round %s", current_round)

        self._set_weights_from_disk("net")

        set_weights(self.vae, parameters[vae_index_start:vae_index_end])

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
            dataset_target_feature=self.dataset_target_feature,
        )

        test_dataloader = dataloader.load_test_dataset_from_disk(
            self.dataset_folder_path, self.batch_size
        )

        results = {"client_number": self.client_number}

        loss, acc = 0.0, 0.0

        if int(current_round) >= self.vae_training_starting_round:
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

            synthetic_dataloader = self._create_synthetic_data()

            # self._save_synthetic_images(synthetic_dataloader, current_round)

            # synthetic_dataloader = self._create_synthetic_data()

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

            torch.save(
                self.net.state_dict(), self.client_model_folder_path + "/model.pth"
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
    vae_epochs = context.run_config.get("vae-epochs")
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
        vae_epochs,
        net_learning_rate,
        dataset_folder_path,
        model_folder_path,
        dataset_input_feature,
        dataset_target_feature,
        samples_per_class,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
