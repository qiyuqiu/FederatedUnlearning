import random
import numpy as np
import torch
from lib_model.MNIST.SimpleCNN import SimpleCNN

num_shards = 20
num_client = 200

# Training parameters
epochs = 100
local_epochs = 1  # Number of local training rounds per client
early_stopping_rounds = 10
lr = 0.001
num_chunk = 10  # Number of chunks to split client training data
client_fraction = 0.2  # Fraction of clients participating per round
sample_fraction = 1  # Fraction of samples used per participating client
tag_is_stop_shard = False  # Whether to stop training a shard early once it's done
tag_is_stop_server = False  # Whether to stop global training early once it's done

# Client clustering strategy
# 1: KMeans
# 2: Hierarchical clustering
# 3: KMeans with dynamic adjustment
# 4: KMeans with enhanced dynamic adjustment
cluster_strategy = 4

# Rank-based training
num_rank = 5

# Pretraining parameters
epochs_pretrain = 1
epochs_client_local = 5
lr_pretrain = lr

# FL + HC (Federated Learning + Hierarchical Clustering)
epochs_pretrain_FL = 10

# Aggregated training parameters
epochs_weights_train_FL = 1
epochs_weights_train = 20
lr_weight_train = 1

# Frequency of dynamic adjustment (i.e., how many rounds between connection training)
epoch_to_adjust = 10

# Whether to enable backtracking in unlearning (saves client models at each participation)
is_save_client_model = False

# Interval for weighted training (i.e., connection training interval)
epochs_weighted_train_interval = 10
num_shard_connection_fraction = 0.5  # Fraction of shards to use in connection training
num_model_use = num_shards * num_shard_connection_fraction

# Communication time per model
time_per_unit = 0.01

# Base of cross-entropy loss
base_cross_loss = 10

sigma = 0.5
BATCH_SIZE = 64

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

cuda_id = 0

if torch.cuda.is_available():
    torch.cuda.set_device(cuda_id)
    device = torch.device('cuda')
    print("Selected GPU:", torch.cuda.get_device_name(0))
    device_index = torch.cuda.current_device()
    print("Current GPU index:", device_index)
else:
    device = torch.device('cpu')


class proj:
    def __init__(self):
        self.num_classes = 10
        self.net_glob = None
        self.lr = lr
        self.device = device
        self.epochs = epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.BATCH_SIZE = BATCH_SIZE
        self.epochs_weights_train = epochs_weights_train
        self.lr_weight_train = lr_weight_train
        self.epochs_pretrain = epochs_pretrain
        self.lr_pretrain = lr_pretrain
        self.num_shards = num_shards
        self.num_model_use = num_model_use
        self.num_client = num_client
        self.epochs_weighted_train_interval = epochs_weighted_train_interval
        self.plot_title = 'None title.png'
        self.epochs_pretrain_FL = epochs_pretrain_FL
        self.epochs_client_local = epochs_client_local
        self.epochs_weights_train_FL = epochs_weights_train_FL
        self.base_cross_loss = base_cross_loss
        self.client_fraction = client_fraction
        self.num_shard_connection_fraction = num_shard_connection_fraction
        self.new_shard_id = self.num_shards + 1
        self.epoch_to_adjust = epoch_to_adjust
        self.sample_fraction = sample_fraction
        self.num_chunk = num_chunk
        self.tag_is_stop_shard = tag_is_stop_shard
        self.tag_is_stop_server = tag_is_stop_server
        self.local_epochs = local_epochs
        self.time_per_unit = time_per_unit
        self.is_save_client_model = is_save_client_model
        self.cluster_strategy = cluster_strategy
        self.num_rank = num_rank
