import json
import os
import time
from typing import List, Tuple

import numpy as np
from flwr.common import (
    Context,
    EvaluateIns,
    FitIns,
    Metrics,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from src.ml_models.cnn import CNN
from src.ml_models.utils import get_weights
from src.scripts.helper import metadata


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
        mode,
        cnn_type,
        num_class_learn_per_round,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_of_clients = num_of_clients
        self.num_server_rounds = num_server_rounds
        self.plots_folder_path = plots_folder_path
        self.dataset_name = dataset_name
        self.mode = mode
        self.cnn_type = cnn_type
        self.num_class_learn_per_round = num_class_learn_per_round

        self.synthetic_data = []
        self.synthetic_labels = []
        self.client_plot = {}
        self.round_times = {}  # Dictionary to store round timing information
        self.round_start_time = None  # Store start time of current round

    def configure_fit(self, server_round, parameters, client_manager):
        # Record start time of this round
        self.round_start_time = time.time()

        # Waiting till all clients are connected
        client_manager.wait_for(self.num_of_clients, timeout=300)

        config: dict[str, Scalar] = {
            "current_round": server_round,
        }

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
        # Calculate and record the time taken in this round
        if self.round_start_time is not None:
            elapsed_time = time.time() - self.round_start_time
            self.round_times[server_round] = elapsed_time

        # Parameters and config
        config: dict[str, Scalar] = {
            "current_round": server_round,
        }

        print("fit_ins.config", config)

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

        if (
            self.mode == "HFedCVAE"
            or self.mode == "HFedCGAN"
            or self.mode == "HFedCVAEGAN"
        ) and server_round <= metadata["num_classes"]:
            if not results:
                return None, {}
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}

            for _, fit_res in results:
                print("fit_res.metrics", fit_res.metrics)
                data = parameters_to_ndarrays(fit_res.parameters)

                if len(data) != 2:
                    continue
                self.synthetic_data.append(data[0])
                self.synthetic_labels.append(data[1])

            return (
                ndarrays_to_parameters(
                    [
                        np.concatenate(self.synthetic_data, axis=0),
                        np.concatenate(self.synthetic_labels, axis=0),
                    ]
                ),
                {},
            )

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
                f"{self.mode}_{self.num_of_clients}_{self.num_class_learn_per_round}_{self.dataset_name}_results.json",
            )

            with open(results_file_path, "w", encoding="utf-8") as file:
                json.dump(self.client_plot, file)

            round_times_file_path = os.path.join(
                self.plots_folder_path,
                f"{self.mode}_{self.num_of_clients}_{self.num_class_learn_per_round}_{self.dataset_name}_round_times.json",
            )

            with open(round_times_file_path, "w", encoding="utf-8") as file:
                json.dump(self.round_times, file)

        return super().aggregate_evaluate(server_round, results, failures)


def server_fn(context: Context):
    # Initialize model parameters
    fraction_evaluate = context.run_config.get("fraction-evaluate")
    num_of_clients = context.run_config.get("num-of-clients")
    num_server_rounds = context.run_config.get("num-server-rounds")
    plots_folder_path = context.run_config.get("plots-folder-path")
    dataset_name = context.run_config.get("dataset-name")
    mode = context.run_config.get("mode")
    cnn_type = context.run_config.get("cnn-type")
    num_class_learn_per_round = context.run_config.get("num-class-learn-per-round")

    ndarrays = get_weights(CNN(cnn_type=str(cnn_type)))
    initial_parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = CustomFedAvg(
        initial_parameters=initial_parameters,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=weighted_average,
        num_of_clients=num_of_clients,
        num_server_rounds=num_server_rounds,
        plots_folder_path=plots_folder_path,
        dataset_name=dataset_name,
        mode=mode,
        cnn_type=cnn_type,
        num_class_learn_per_round=num_class_learn_per_round,
    )
    config = ServerConfig(num_rounds=int(context.run_config["num-server-rounds"]))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
