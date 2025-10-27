from lib_cluster.connection_avg_aggregate import aggregate_connection


class Shard:
    def __init__(self, shard_id, proj):
        self.proj = proj
        self.shard_id = shard_id
        self.model = None  # Model of the current shard
        self.clients = None  # Clients contained in the current shard
        self.iswork = True  # Indicates whether this shard is active
        self.is_trained_completely = False  # Indicates whether the shard has finished training
        self.early_stopping_rounds = proj.early_stopping_rounds

        self.num_connected = proj.num_shards - 1
        self.shard_connections = None  # Weights of all shard connections on the server side

        # Unlearning priority rank of the shard; higher rank means easier to be unlearned
        self.rank = 0
        self.saved_model_rank = None
        self.ranks = []

        # Lists for storing data from each training round
        self.loss_train_epoch = []
        self.loss_test_epoch = []
        self.acc_train_epoch = []
        self.acc_test_epoch = []

    def eval_is_stop_train(self):
        if len(self.loss_test_epoch) < self.early_stopping_rounds:
            return
        min_test_loss = min(self.loss_test_epoch)
        min_loss_index = self.loss_test_epoch.index(min_test_loss)
        steps_since_min_loss = len(self.loss_test_epoch) - min_loss_index - 1
        if steps_since_min_loss >= self.early_stopping_rounds:
            self.is_trained_completely = True
            for client in self.clients:
                client.is_trained_completely = True
            print(f'Shard {self.shard_id} - No improvement in test loss for {self.early_stopping_rounds} '
                  f'consecutive rounds after the minimum test loss. Stopping early.')

    def train_weights_FL(self):
        self.broadcast_shard_connection()
        for client in self.clients:
            client.train_weights()
        self.aggregate_weights()

    def broadcast_model(self):
        for client in self.clients:
            client.receive_model(self.model.state_dict())

    # Send weights and models in shard_connections to each client
    def broadcast_shard_connection(self):
        for client in self.clients:
            client.receive_shard_connection(self.shard_connections)

    # Receive all client-uploaded shard connection weights and perform weighted aggregation
    def aggregate_weights(self):
        shard_connections_list = [client.upload_shard_connection_weight() for client in self.clients]
        shard_connections_avg = aggregate_connection(shard_connections_list)
        for sc_avg in shard_connections_avg:
            for sc in self.shard_connections:
                if sc_avg.shard_id == sc.shard_id:
                    sc.weight = sc_avg.weight
