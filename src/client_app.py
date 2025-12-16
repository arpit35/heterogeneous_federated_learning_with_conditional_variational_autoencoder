from flwr.client import ClientApp
from flwr.common import Context

from src.clients.flower_h_fed_pfs_client import FlowerHFedPFSClient
from src.clients.flower_vae_client import FlowerVAEClient


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
    mode = context.run_config.get("mode")

    if mode == "vqe":
        return FlowerVAEClient(
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
    elif mode == "HFedPFS":
        return FlowerHFedPFSClient(
            client_number,
            batch_size,
            net_epochs,
            net_learning_rate,
            dataset_folder_path,
            model_folder_path,
            dataset_input_feature,
            dataset_target_feature,
        ).to_client()
    else:
        raise ValueError(
            f"Unsupported mode: {mode}. This client only supports 'vqe' mode."
        )


# Flower ClientApp
app = ClientApp(client_fn)
