import os
import random
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import transforms

from src.scripts.helper import save_metadata


class NumpyDataset(Dataset):
    def __init__(
        self,
        images,
        labels,
        dataset_input_feature: str,
        dataset_target_feature: str,
    ):
        """
        images: np.ndarray, shape [N, H, W] or [N, C, H, W]
        labels: np.ndarray or list, shape [N]
        """
        self.images = images
        self.labels = labels
        self.dataset_input_feature = dataset_input_feature
        self.dataset_target_feature = dataset_target_feature

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # ⚠️ ALWAYS allocates new memory
        x = torch.tensor(img, dtype=torch.float32)

        y = torch.tensor(label, dtype=torch.long)

        return {
            self.dataset_input_feature: x,
            self.dataset_target_feature: y,
        }


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
        self.pytorch_transforms = self.pytorch_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def _apply_transforms(self, batch):
        if batch.get(self.dataset_input_feature):
            transformed_images = []
            for img in batch[self.dataset_input_feature]:
                if isinstance(img, list):
                    transformed_images.append(torch.tensor(img, dtype=torch.float32))
                else:
                    transformed_images.append(self.pytorch_transforms(img))
            batch[self.dataset_input_feature] = transformed_images
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
        }

    def limit_samples_per_class(self, dataset, max_per_class, seed=42):
        random.seed(seed)

        class_indices = defaultdict(list)

        # Collect indices per class
        for idx, example in enumerate(dataset):
            cls = example[self.dataset_target_feature]
            class_indices[cls].append(idx)

        # Subsample indices
        selected_indices = []
        for cls, indices in class_indices.items():
            if len(indices) > max_per_class:
                indices = random.sample(indices, max_per_class)
            selected_indices.extend(indices)

        # Create filtered dataset
        return dataset.select(selected_indices)

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

            partition = self.limit_samples_per_class(partition, max_per_class=6000)

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

            partition_train_val = partition.train_test_split(
                test_size=0.2, seed=42, stratify_by_column=self.dataset_target_feature
            )

            train_partition = partition_train_val["train"]
            val_partition = partition_train_val["test"]

            train_path = os.path.join(client_dataset_folder_path, "train_data")
            val_path = os.path.join(client_dataset_folder_path, "val_data")

            train_partition.save_to_disk(train_path)
            val_partition.save_to_disk(val_path)

    def _process_target_classes(self, dataset, target_classes, upsample_amount):
        """
        Process multiple target classes in a single pass through the dataset.
        """
        # Validate inputs
        if not target_classes:
            return None, {}

        # Initialize data structures
        all_target_class_num_samples = {cls: 0 for cls in target_classes}
        class_data = {cls: [] for cls in target_classes}

        # Single pass through dataset
        for example in dataset:
            current_class = example[self.dataset_target_feature]
            if current_class in all_target_class_num_samples:
                all_target_class_num_samples[current_class] += 1
                class_data[current_class].append(example)

        # Check for classes with insufficient samples
        filtered_target_class_num_samples = {}
        filtered_and_upscaled_data = []

        for cls in target_classes:
            num_samples = all_target_class_num_samples[cls]

            if num_samples < 100:
                continue  # Skip this class

            # Add original samples
            samples = class_data[cls]
            filtered_target_class_num_samples[cls] = num_samples
            filtered_and_upscaled_data.extend(samples)

            # Apply upsampling if needed
            if num_samples < upsample_amount and upsample_amount > 0:
                # Generate extra indices with replacement
                extra_indices = np.random.choice(
                    num_samples, upsample_amount - num_samples, replace=True
                )
                # Add extra samples
                filtered_and_upscaled_data.extend(samples[i] for i in extra_indices)

        return filtered_and_upscaled_data, filtered_target_class_num_samples

    def load_dataset_from_disk(
        self,
        data_type: str,
        client_folder_path: str,
        batch_size: int,
        target_class: list | None = None,
        upsample_amount: int = 0,
    ) -> TorchDataLoader | None | tuple[TorchDataLoader, dict] | tuple[None, dict]:

        client_file_path = os.path.join(client_folder_path, data_type)
        dataset = load_from_disk(client_file_path)

        # ---- 1) Filter and process target classes in single pass ----
        if target_class is not None:
            filtered_and_upscaled_data, filtered_target_class_num_samples = (
                self._process_target_classes(dataset, target_class, upsample_amount)
            )

            if not filtered_target_class_num_samples:
                # Count all classes for the return value
                all_target_class_num_samples = {}
                for cls in target_class:
                    # If we need accurate counts without filtering, we can either:
                    # Option 1: Use a faster counting method
                    all_target_class_num_samples[cls] = sum(
                        1
                        for example in dataset
                        if example[self.dataset_target_feature] == cls
                    )
                    # Option 2: Use the counts from the single pass (already computed)
                    # all_target_class_num_samples[cls] = 0  # We'd need to track this
                return None, all_target_class_num_samples

            dataset = dataset.from_list(filtered_and_upscaled_data)

        else:
            num_samples = len(dataset)
            if num_samples < 100:
                return None

        # Apply transforms
        dataset = dataset.with_transform(self._apply_transforms)

        # Return dataloader
        dataloader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        if target_class is not None:
            return dataloader, filtered_target_class_num_samples

        return dataloader

    def load_dataset_from_ndarray(self, parameters, batch_size) -> TorchDataLoader:
        dataset = NumpyDataset(
            images=parameters[0],
            labels=parameters[1],
            dataset_input_feature=self.dataset_input_feature,
            dataset_target_feature=self.dataset_target_feature,
        )

        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
