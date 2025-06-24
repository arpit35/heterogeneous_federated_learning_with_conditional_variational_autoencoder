import os

import numpy as np
from datasets import load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import transforms

from src.scripts.helper import metadata, save_metadata


class DataLoader:
    def __init__(
        self,
        dataset_input_feature: str,
        dataset_target_feature: str = "label",
        dataset_name: str = "mnist",
    ):
        self.dataset_name = dataset_name
        self.dataset_input_feature = dataset_input_feature
        self.dataset_target_feature = dataset_target_feature
        self.pytorch_transforms = None

    def _apply_transforms(self, batch):
        if self.pytorch_transforms is None:
            num_channels = metadata.get("num_channels")

            if num_channels == 1:
                normalize = transforms.Normalize((0.5,), (0.5,))
            elif num_channels == 3:
                normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            else:
                raise ValueError(f"Unsupported number of channels: {num_channels}")

            self.pytorch_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            )

        batch[self.dataset_input_feature] = [
            self.pytorch_transforms(img) for img in batch[self.dataset_input_feature]
        ]
        return batch

    def _load_partition(self, num_clients: int, alpha: float):
        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by=self.dataset_target_feature,
            alpha=alpha,
            self_balancing=True,
        )
        return FederatedDataset(
            dataset=self.dataset_name, partitioners={"train": partitioner}
        )

    def _get_dataset_metadata(self, fds: FederatedDataset, num_clients: int):
        # Take one full partition to extract dataset-wide metadata
        sample_partition = fds.load_partition(0)
        sample_example = sample_partition[0]

        # Get image shape and channels
        image = sample_example[self.dataset_input_feature]
        print(f"Sample image type: {type(image)}")

        if hasattr(image, "shape"):
            height, width = image.shape[:2]
            channels = 1 if len(image.shape) == 2 else image.shape[2]
        elif hasattr(image, "size"):  # PIL Image
            width, height = image.size
            # Infer channels from mode
            mode_to_channels = {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4}
            channels = mode_to_channels.get(image.mode, 1)
        else:
            raise ValueError("Cannot determine image dimensions from sample.")

        # Compute number of classes
        all_labels = []
        for client_id in range(num_clients):
            partition = fds.load_partition(client_id)
            all_labels.extend(partition[self.dataset_target_feature])

        num_classes = len(set(all_labels))

        # Save metadata
        return {
            "num_classes": num_classes,
            "image_width": width,
            "image_height": height,
            "num_channels": channels,
            "encoder_hidden_dim": int((width * height * channels) / 10),
            "generic_encoder_latent_dim": int((width + height + channels) / 5),
            "personalized_encoder_latent_dim": int((width + height + channels) / 10),
            "decoder_hidden_dim": int((width * height * channels) / 10),
        }

    def save_datasets_to_disk(
        self,
        num_clients: int,
        alpha: float,
        dataset_folder_path: str,
    ):
        fds = self._load_partition(num_clients, alpha)

        # Get dataset metadata
        save_metadata(self._get_dataset_metadata(fds, num_clients))

        for client_id in range(num_clients):
            client_dataset_folder_path = os.path.join(
                dataset_folder_path, f"client_{client_id}"
            )
            os.makedirs(client_dataset_folder_path, exist_ok=True)

            partition = fds.load_partition(client_id)

            labels = partition[self.dataset_target_feature]
            # Compute the unique classes and their counts
            unique_classes, counts = np.unique(labels, return_counts=True)

            # Filter out classes with only one row
            classes_to_keep = set(unique_classes[counts > 1])
            partition = partition.filter(
                lambda example, keep=classes_to_keep: example[
                    self.dataset_target_feature
                ]
                in keep,
                load_from_cache_file=False,
            )

            partition_train_test = partition.train_test_split(
                test_size=0.2, seed=42, stratify_by_column=self.dataset_target_feature
            )

            train_partition = partition_train_test["train"]
            test_partition = partition_train_test["test"]

            train_path = os.path.join(client_dataset_folder_path, "train_data")
            test_path = os.path.join(client_dataset_folder_path, "test_data")

            train_partition.save_to_disk(train_path)
            test_partition.save_to_disk(test_path)

    def load_dataset_from_disk(
        self,
        data_type: str,
        client_folder_path,
        batch_size,
    ) -> TorchDataLoader:
        client_file_path = os.path.join(client_folder_path, data_type)

        client_dataset = load_from_disk(client_file_path).with_transform(
            self._apply_transforms
        )

        data_loader = TorchDataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        return data_loader
