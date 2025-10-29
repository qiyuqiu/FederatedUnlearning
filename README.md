# EFU

This repository contains the implementation of our experiments, which include reproductions of several existing unlearning methods. We provide a detailed comparison with the following approaches: FLB, FL+HC, FATS, FedCIO, and EFU. Each of these methods can be executed individually using the respective scripts (`run+work`), or all can be executed together using the `run_all` script. Detailed descriptions of the comparison strategies can be found in the accompanying paper.

## Usage

### Data Preparation
The data loading code is already implemented. To use the code, please follow these steps:

1. **Set Data Storage Paths:**
   - Fill in the paths for the datasets in the `lib_gate` file. Specifically, locate the corresponding data storage paths in the dataset configuration files.

2. **Automatic Dataset Download:**
   - The MNIST and FMNIST datasets will be automatically downloaded.
   - For CIFAR-10, the download link is available on the official website.
   - The FEMNIST dataset can be downloaded using the Leaf code.

3. **Set Environment Path:**
   - Ensure you specify your environment path in the `run_all` script.

4. **Adjust Experiment Parameters:**
   - Parameters relevant to the experiments can be modified in the `attribute_FL` file. Each variable is accompanied by comments for clarity. Readers can adjust these parameters according to their needs and then run the tests.

# Key parameters
------------------------------------------------------------------------------

- num_shards (int): Number of data shards used to partition datasets across clients.
- num_client (int): Total number of clients involved in the federated training.
- epochs (int): Number of global communication rounds between clients and the server.
- local_epochs (int): Number of local training epochs per client in each round.
- early_stopping_rounds (int): Early stopping is triggered if no improvement is observed 
  over this many communication rounds.
- lr (float): Learning rate for global model updates.
- client_fraction (float): Fraction of clients selected randomly in each round to participate.
- epochs_pretrain (int): Number of communication rounds used for initial pretraining.
- epochs_client_local (int): Local training epochs per client during pretraining.
- lr_pretrain (float): Learning rate for pretraining; usually the same as lr.
- epochs_weights_train (int): Number of communication rounds to train inter-shard similarity weights.
- lr_weight_train (float): Learning rate used for connection weight optimization.
- cluster_strategy (int): 
    1 = KMeans-based clustering
    2 = Hierarchical clustering
    4 = Dynamic strategy (adaptive merging and splitting)
- BATCH_SIZE (int): Batch size used during local client training.
- SEED (int): Random seed for reproducibility.

## Software Requirements

- **PyTorch**: 1.9.0
- **CUDA**: 11.1

The required software libraries are listed in the `requirements.txt` file. Install them using:

```bash
pip install -r requirements.txt
