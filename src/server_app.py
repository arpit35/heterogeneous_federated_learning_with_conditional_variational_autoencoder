import json
import os
from typing import List, Tuple

from flwr.common import Context, EvaluateIns, FitIns, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from src.ml_models.net import get_net
from src.ml_models.utils import get_weights
from src.server_components.aggregate_evaluate import (
    aggregate_evaluate_federated_unlearning,
)
from src.server_components.aggregate_fit import (
    aggregate_fit_federated_learning,
    aggregate_fit_federated_unlearning,
    aggregate_fit_retraining,
)
from src.server_components.configure_evaluate import (
    configure_evaluate_federated_learning,
    configure_evaluate_federated_unlearning,
)
from src.server_components.configure_fit import (
    configure_fit_federated_learning,
    configure_fit_federated_unlearning,
)
from src.server_components.helper import save_model_to_disk
from src.server_components.server_init import server_init_federated_unlearning


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


class CustomFedAvg(FedAvg):
    def __init__(
        self,
        mode,
        num_of_clients,
        num_server_rounds,
        weight_factor_global_model,
        model_name,
        dataset_name,
        plots_folder_path,
        models_folder_path,
        knowledge_eraser_rounds,
        command,
        global_model_parameters,
        degraded_model_parameters,
        unlearning_trigger_client,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.unlearn_client_number = unlearning_trigger_client
        self.unlearn_client_id = -1
        self.command = command
        self.current_knowledge_eraser_round = 1

        self.mode = mode
        self.num_of_clients = num_of_clients
        self.num_server_rounds = num_server_rounds
        self.weight_factor_global_model = weight_factor_global_model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.plots_folder_path = plots_folder_path
        self.models_folder_path = models_folder_path
        self.knowledge_eraser_rounds = knowledge_eraser_rounds

        self.degraded_model_parameters = degraded_model_parameters
        self.global_model_parameters = global_model_parameters
        self.client_plot = {}

    def configure_fit(self, server_round, parameters, client_manager):
        # Waiting till all clients are connected
        client_manager.wait_for(self.num_of_clients)

        config = {
            "current_round": server_round,
            "unlearn_client_number": self.unlearn_client_number,
            "command": self.command,
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

        if self.mode == "federated_learning" or self.mode == "retraining":
            return configure_fit_federated_learning(fit_ins, clients)
        elif self.mode == "federated_unlearning":
            return configure_fit_federated_unlearning(self, fit_ins, clients, config)

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Parameters and config
        config = {
            "current_round": server_round,
            "unlearn_client_number": self.unlearn_client_number,
            "command": self.command,
        }

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if self.mode == "federated_learning" or self.mode == "retraining":
            return configure_evaluate_federated_learning(evaluate_ins, clients)
        elif self.mode == "federated_unlearning":
            return configure_evaluate_federated_unlearning(
                self, evaluate_ins, clients, config
            )

    def aggregate_fit(self, server_round, results, failures):

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.mode == "federated_learning":
            return aggregate_fit_federated_learning(self, results)
        elif self.mode == "retraining":
            return aggregate_fit_retraining(results)
        elif self.mode == "federated_unlearning":
            return aggregate_fit_federated_unlearning(self, server_round, results)

    def aggregate_evaluate(self, server_round, results, failures):

        for _, eval_res in results:
            client_number = eval_res.metrics["client_number"]

            if client_number not in self.client_plot:
                self.client_plot[client_number] = {}

            self.client_plot[client_number][server_round] = {
                "metrics": json.loads(str(eval_res.metrics).replace("'", '"')),
                "command": self.command,
            }

        if server_round == self.num_server_rounds:
            results_file_path = os.path.join(
                self.plots_folder_path,
                self.mode,
                f"{self.dataset_name}_{self.model_name}_results.json",
            )

            with open(results_file_path, "w", encoding="utf-8") as file:
                json.dump(self.client_plot, file)

            if self.mode == "federated_learning":
                save_model_to_disk(self)

        if self.mode == "federated_unlearning":
            aggregate_evaluate_federated_unlearning(self)

        return super().aggregate_evaluate(server_round, results, failures)


def server_fn(context: Context):
    print("context.node_config", context)
    # Initialize model parameters
    mode = context.run_config.get("mode", None)
    dataset_num_channels = context.run_config.get("dataset-num-channels", None)
    dataset_num_classes = context.run_config.get("dataset-num-classes", None)
    model_name = context.run_config.get("model-name", None)
    dataset_name = context.run_config.get("dataset-name", None)
    plots_folder_path = context.run_config.get("plots-folder-path", None)
    models_folder_path = context.run_config.get("models-folder-path", None)
    fraction_evaluate = context.run_config.get("fraction-evaluate", None)
    num_of_clients = context.run_config.get("num-of-clients", None)
    num_server_rounds = context.run_config.get("num-server-rounds", None)
    weight_factor_degradation_model = context.run_config.get(
        "weight-factor-degradation-model", None
    )
    weight_factor_global_model = context.run_config.get(
        "weight-factor-global-model", None
    )
    knowledge_eraser_rounds = context.run_config.get("knowledge-eraser-rounds", None)
    unlearning_trigger_client = context.run_config.get("unlearning-trigger-client", -1)

    ndarrays = get_weights(
        get_net(dataset_num_channels, dataset_num_classes, model_name)
    )

    parameters = None
    command = ""
    global_model_parameters = None
    degraded_model_parameters = None

    if mode == "federated_learning" or mode == "retraining":
        parameters = ndarrays_to_parameters(ndarrays)
    elif mode == "federated_unlearning":
        command, parameters, global_model_parameters, degraded_model_parameters = (
            server_init_federated_unlearning(
                models_folder_path,
                dataset_name,
                model_name,
                dataset_num_channels,
                dataset_num_classes,
                weight_factor_degradation_model,
            )
        )

    # Define the strategy
    strategy = CustomFedAvg(
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        mode=mode,
        num_of_clients=num_of_clients,
        num_server_rounds=num_server_rounds,
        weight_factor_global_model=weight_factor_global_model,
        model_name=model_name,
        dataset_name=dataset_name,
        plots_folder_path=plots_folder_path,
        models_folder_path=models_folder_path,
        knowledge_eraser_rounds=knowledge_eraser_rounds,
        command=command,
        global_model_parameters=global_model_parameters,
        degraded_model_parameters=degraded_model_parameters,
        unlearning_trigger_client=unlearning_trigger_client,
    )
    config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
