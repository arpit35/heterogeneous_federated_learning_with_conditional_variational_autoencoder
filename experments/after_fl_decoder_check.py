import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torchvision import transforms

# Add the root directory to the path to import the model
root_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_dir)
from src.ml_models.HFedCVAE import HFedCVAE


class HuggingFaceDatasetWrapper(Dataset):
    """Wrapper to make HuggingFace Dataset compatible with PyTorch DataLoader"""

    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        return self.hf_dataset[idx]


class ImageProcessor:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.dataset_input_feature = "image"

        # Initialize transforms
        self.pytorch_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # Load dataset
        self.dataset = self.load_dataset()

        # Load model
        self.model = self.load_model()

    def load_dataset(self):
        """Load the dataset from the specified path"""
        try:
            dataset = load_from_disk(self.data_path).with_transform(
                self._apply_transforms
            )
            print(f"Dataset loaded successfully with {len(dataset)} samples")
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def _apply_transforms(self, batch):
        """Apply transforms to the batch"""
        batch[self.dataset_input_feature] = [
            self.pytorch_transforms(img) for img in batch[self.dataset_input_feature]
        ]
        return batch

    def load_model(self):
        """Load the trained model"""
        try:
            # Try different CNN types to find the one that matches the saved model
            cnn_types = ["cnn1", "cnn2", "cnn3", "cnn4"]

            for cnn_type in cnn_types:
                try:
                    print(f"Trying to load model with {cnn_type} configuration...")
                    model = HFedCVAE(cnn_type=cnn_type)

                    # Load the trained weights
                    state_dict = torch.load(self.model_path, map_location="cpu")
                    model.load_state_dict(state_dict)
                    model.eval()

                    print(f"Model loaded successfully with {cnn_type} configuration")
                    return model

                except Exception as e:
                    print(f"Failed to load with {cnn_type}: {str(e)[:100]}...")
                    continue

            print("Could not load model with any CNN configuration")
            return None

        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def generate_and_plot_images(self, num_samples=5):
        """Generate and plot images using the loaded model"""
        if self.dataset is None or self.model is None:
            print("Dataset or model not loaded properly")
            return

        # Create data loader using the wrapper
        wrapped_dataset = HuggingFaceDatasetWrapper(self.dataset)
        data_loader = TorchDataLoader(
            wrapped_dataset,
            batch_size=num_samples,
            shuffle=True,
        )

        # Get a batch of images
        batch = next(iter(data_loader))
        images = batch[self.dataset_input_feature]
        labels = batch.get("label", None)

        # Convert to tensor if needed
        if not isinstance(images, torch.Tensor):
            images = torch.stack(images)

        print(f"Processing {images.shape[0]} images with shape {images.shape}")

        with torch.no_grad():
            # Forward pass through the model
            try:
                # Use the model's forward method which handles the complete pipeline
                y_logits, recon_x, mu_z, logvar_z, mu_c, logvar_c = self.model(images)

                # Get classification predictions
                predicted_classes = torch.argmax(y_logits, dim=1)

                # The reconstruction is already done by the model
                reconstructed = recon_x

                print("Model forward pass successful")

            except Exception as e:
                print(f"Error during model forward pass: {e}")
                return

        # Plot original and reconstructed images
        fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))

        for i in range(num_samples):
            # Original image
            original_img = images[i].squeeze().numpy()
            axes[0, i].imshow(original_img, cmap="gray")
            axes[0, i].set_title(f"Original")
            axes[0, i].axis("off")

            # Reconstructed image
            recon_img = reconstructed[i].reshape(28, 28).numpy()
            axes[1, i].imshow(recon_img, cmap="gray")
            axes[1, i].set_title(f"Reconstructed")
            axes[1, i].axis("off")

            # Difference
            diff_img = np.abs(original_img - recon_img)
            axes[2, i].imshow(diff_img, cmap="hot")
            axes[2, i].set_title(f"Difference")
            axes[2, i].axis("off")

            # Add classification result if available
            if labels is not None:
                true_label = labels[i].item()
                pred_label = predicted_classes[i].item()
                axes[0, i].set_title(f"Original (True: {true_label})")
                axes[1, i].set_title(f"Reconstructed (Pred: {pred_label})")

        plt.suptitle("Original vs Reconstructed Images")
        plt.tight_layout()

        # Save the plot
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, "reconstruction_comparison.png"),
            dpi=150,
            bbox_inches="tight",
        )
        print(
            f"Reconstruction comparison saved to {output_dir}/reconstruction_comparison.png"
        )
        plt.show()

        # Plot latent space visualization
        # Sample from the latent distributions for visualization
        z = self.model.reparameterize(mu_z, logvar_z)
        c = self.model.reparameterize(mu_c, logvar_c)
        self.plot_latent_space(z, c, labels)

    def plot_latent_space(self, z, c, labels=None):
        """Plot latent space representations"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        print(f"Latent space dimensions - z: {z.shape}, c: {c.shape}")

        # Plot generic encoder latent space (z)
        z_np = z.numpy()
        if z_np.shape[1] >= 2:
            scatter = axes[0].scatter(
                z_np[:, 0],
                z_np[:, 1],
                c=labels.numpy() if labels is not None else "blue",
                cmap="tab10",
                alpha=0.7,
            )
            axes[0].set_title("Generic Encoder Latent Space (z)")
            axes[0].set_xlabel("z_0")
            axes[0].set_ylabel("z_1")
            if labels is not None:
                plt.colorbar(scatter, ax=axes[0])

        # Plot personalized encoder latent space (c)
        c_np = c.numpy()
        if c_np.shape[1] >= 2:
            scatter = axes[1].scatter(
                c_np[:, 0],
                c_np[:, 1],
                c=labels.numpy() if labels is not None else "red",
                cmap="tab10",
                alpha=0.7,
            )
            axes[1].set_title("Personalized Encoder Latent Space (c)")
            axes[1].set_xlabel("c_0")
            axes[1].set_ylabel("c_1")
            if labels is not None:
                plt.colorbar(scatter, ax=axes[1])

        plt.tight_layout()

        # Save the latent space plot
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, "latent_space_visualization.png"),
            dpi=150,
            bbox_inches="tight",
        )
        print(
            f"Latent space visualization saved to {output_dir}/latent_space_visualization.png"
        )
        plt.show()


def main():
    # Define paths
    data_path = "src/clients_dataset/client_4/train_data"
    model_path = "src/clients_models/client_4/model.pth"

    # Make paths relative to the project root, not the script location
    project_root = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(project_root, data_path)
    model_path = os.path.join(project_root, model_path)

    print(f"Loading data from: {data_path}")
    print(f"Loading model from: {model_path}")

    # Create processor and run
    processor = ImageProcessor(data_path, model_path)
    processor.generate_and_plot_images(num_samples=8)


if __name__ == "__main__":
    main()
