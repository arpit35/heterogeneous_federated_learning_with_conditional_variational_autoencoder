import json
import os
from typing import List, Tuple

import numpy as np
from flwr.common import (
    Context,
    EvaluateIns,
    FitIns,
    Metrics,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from src.ml_models.decoder import Decoder
from src.ml_models.generic_encoder import GenericEncoder
from src.ml_models.personalized_encoder import PersonalizedEncoder
from src.ml_models.utils import get_weights


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [
        float(num_examples) * float(m["accuracy"])
        for num_examples, m in metrics
        if isinstance(m.get("accuracy", None), (int, float))
    ]
    examples = [
        float(num_examples)
        for num_examples, m in metrics
        if isinstance(m.get("accuracy", None), (int, float))
    ]
    return {"accuracy": sum(accuracies) / sum(examples) if examples else 0.0}


class CustomFedAvg(FedAvg):
    def __init__(
        self,
        num_of_clients,
        num_server_rounds,
        plots_folder_path,
        dataset_name,
        parameter_indices_config,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_of_clients = num_of_clients
        self.num_server_rounds = num_server_rounds
        self.plots_folder_path = plots_folder_path
        self.dataset_name = dataset_name
        self.parameter_indices_config = parameter_indices_config
        self.client_plot = {}

    def configure_fit(self, server_round, parameters, client_manager):
        # Waiting till all clients are connected
        client_manager.wait_for(self.num_of_clients, timeout=300)

        config: dict[str, Scalar] = {
            "current_round": server_round,
        }

        if server_round == 1:
            # Initialize the parameters for the first round
            config = {**config, **self.parameter_indices_config}

        print("fit_ins.config", config)

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, fit_ins) for client in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Parameters and config
        config: dict[str, Scalar] = {
            "current_round": server_round,
        }

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        for _, fit_res in results:
            print("fit_res.metrics", fit_res.metrics)
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):

        for _, eval_res in results:
            print("eval_res.metrics", eval_res.metrics)
            client_number = eval_res.metrics["client_number"]

            if client_number not in self.client_plot:
                self.client_plot[client_number] = {}

            self.client_plot[client_number][server_round] = {
                "metrics": json.loads(str(eval_res.metrics).replace("'", '"')),
            }

        if server_round == self.num_server_rounds:
            os.makedirs(self.plots_folder_path, exist_ok=True)

            results_file_path = os.path.join(
                self.plots_folder_path,
                f"{self.dataset_name}_results.json",
            )

            with open(results_file_path, "w", encoding="utf-8") as file:
                json.dump(self.client_plot, file)

        return super().aggregate_evaluate(server_round, results, failures)


def server_fn(context: Context):
    # Initialize model parameters
    fraction_evaluate = context.run_config.get("fraction-evaluate")
    num_of_clients = context.run_config.get("num-of-clients")
    num_server_rounds = context.run_config.get("num-server-rounds")
    plots_folder_path = context.run_config.get("plots-folder-path")
    dataset_name = context.run_config.get("dataset-name")

    generic_encoder_parameters = get_weights(GenericEncoder())
    personalized_encoder_parameters = get_weights(PersonalizedEncoder())
    decoder_parameters = get_weights(Decoder())

    all_weights = (
        generic_encoder_parameters
        + personalized_encoder_parameters
        + decoder_parameters
    )

    # Convert to a single Parameters object
    initial_parameters = ndarrays_to_parameters(all_weights)

    # Calculate parameter indices
    gen_enc_start = 0
    gen_enc_end = len(generic_encoder_parameters)
    pers_enc_start = gen_enc_end
    pers_enc_end = pers_enc_start + len(personalized_encoder_parameters)
    dec_start = pers_enc_end
    dec_end = dec_start + len(decoder_parameters)

    parameter_indices_config = {
        "generic_encoder_start": gen_enc_start,
        "generic_encoder_end": gen_enc_end,
        "personalized_encoder_start": pers_enc_start,
        "personalized_encoder_end": pers_enc_end,
        "decoder_start": dec_start,
        "decoder_end": dec_end,
    }

    # Define the strategy
    strategy = CustomFedAvg(
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=weighted_average,
        num_of_clients=num_of_clients,
        num_server_rounds=num_server_rounds,
        plots_folder_path=plots_folder_path,
        dataset_name=dataset_name,
        parameter_indices_config=parameter_indices_config,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=int(context.run_config["num-server-rounds"]))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
