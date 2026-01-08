from flwr.client import ClientApp
from flwr.common import Context

from src.clients.flower_default_client import FlowerDefaultClient
from src.clients.flower_h_fed_cvae_client import FlowerHFedCVAEClient


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
    mode = context.run_config.get("mode")
    cnn_type = context.run_config.get("cnn-type")
    num_class_learn_per_round = context.run_config.get("num-class-learn-per-round")
    kl_loss_beta = context.run_config.get("kl-loss-beta")

    if mode == "HFedCVAE" or mode == "HFedCGAN" or mode == "HFedCVAEGAN":
        return FlowerHFedCVAEClient(
            client_number,
            batch_size,
            net_epochs,
            vae_epochs,
            net_learning_rate,
            dataset_folder_path,
            model_folder_path,
            dataset_input_feature,
            dataset_target_feature,
            cnn_type,
            mode,
            num_class_learn_per_round,
            kl_loss_beta,
        ).to_client()
    elif mode == "default":
        return FlowerDefaultClient(
            client_number,
            batch_size,
            net_epochs,
            net_learning_rate,
            dataset_folder_path,
            model_folder_path,
            dataset_input_feature,
            dataset_target_feature,
            cnn_type,
        ).to_client()
    else:
        raise ValueError(
            f"Unsupported mode: {mode}. This client only supports 'vqe' mode."
        )


# Flower ClientApp
app = ClientApp(client_fn)
