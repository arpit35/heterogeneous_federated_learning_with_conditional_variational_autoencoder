# heterogeneous_federated_learning_with_disentangled_variational_autoencoder

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```


## Running the Application

To prepare the dataset and clear logs before running the Flower application, use the following command:

```sh
python pre_flwr_run.py
```

To run the Flower application, use the following command:

```sh
flwr run .
```

## Model Component Ratios

- GAN:     D:G = 1.5:1
- VAE:     E:G = 1:1
- VAE-GAN: E:G:D = 1:1:0.8