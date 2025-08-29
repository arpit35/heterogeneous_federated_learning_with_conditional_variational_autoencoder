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
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_of_clients = num_of_clients
        self.num_server_rounds = num_server_rounds
        self.plots_folder_path = plots_folder_path
        self.dataset_name = dataset_name
        self.client_plot = {}

    def configure_fit(self, server_round, parameters, client_manager):
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
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Collect data from all clients
        all_client_data = []

        for _, fit_res in results:
            print("fit_res.metrics", fit_res.metrics)
            synthetic_data = parameters_to_ndarrays(fit_res.parameters)
            all_client_data.append(synthetic_data)

        if all_client_data:
            num_arrays = len(all_client_data[0])

            aggregated_ndarrays = []

            for array_idx in range(num_arrays):
                arrays_at_position = [
                    client_data[array_idx] for client_data in all_client_data
                ]

                print("array_idx", array_idx, len(arrays_at_position))

                concatenated_array = np.concatenate(arrays_at_position, axis=0)
                aggregated_ndarrays.append(concatenated_array)

            parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

            return parameters_aggregated, {}

        return None, {}

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

    # Define the strategy
    strategy = CustomFedAvg(
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=weighted_average,
        num_of_clients=num_of_clients,
        num_server_rounds=num_server_rounds,
        plots_folder_path=plots_folder_path,
        dataset_name=dataset_name,
    )
    config = ServerConfig(num_rounds=int(context.run_config["num-server-rounds"]))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
