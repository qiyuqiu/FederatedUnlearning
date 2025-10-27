import copy
import random
import gc
from lib_client.ClientLearn import LocalUpdate
from lib_connection_train.WeightTrain_all import Shard_Weights_trainer
from lib_getdata.DataSplite import DatasetSplit
from lib_utils.eval import predict_with_shards
from lib_utils.untils import *
from lib_connection_train.Shard_connection import *
from torch.utils.data import DataLoader


class Client:
    def __init__(self, client_id, data_train, data_test, project):
        self.proj = project
        self.iswork = True
        self.is_trained_completely = False
        self.data_test = data_test
        self.data_train = data_train
        self.model = None
        self.shard = 0
        self.local_epoch = self.proj.local_epochs
        self.client_id = client_id
        self.model_before_train = None
        self.unlearn_prob = 0
        self.rank = None
        self.shard_connections = None
        if len(data_train) == 0:
            self.iswork = 0
        self.loss_train = None
        self.acc_train = None
        self.loss_test = None
        self.acc_test = None
        self.acc_test_shard = None

        self.data_chunks_indices = split_data_into_chunks(self.data_train, self.proj.num_chunk)

    # local train
    def local_update(self):
        if self.proj.is_save_client_model:
            if self.model_before_train is None:
                self.model_before_train = copy.deepcopy(self.model)
                self.model_before_train.to('cpu')
        if len(self.data_train) == 0:
            return

        sample_size = int(self.proj.num_chunk * self.proj.sample_fraction)
        selected_indices = random.sample(range(self.proj.num_chunk), sample_size)
        selected_data_indices = [idx for i in selected_indices for idx in self.data_chunks_indices[i]]
        selected_data_train = DatasetSplit(self.data_train, selected_data_indices)
        client_trainer = LocalUpdate(self.client_id, self.proj.lr, self.proj.device,
                                     self.proj.BATCH_SIZE, selected_data_train, self.data_test)
        client_trainer.local_ep = self.local_epoch
        self.model.to(self.proj.device)
        loss_test, acc_test = client_trainer.evaluate(self.model)
        w, loss_train, acc_train = client_trainer.train(self.model)
        net = copy.deepcopy(self.model)
        net.load_state_dict(w)
        self.model = net
        self.model.to('cpu')
        torch.cuda.empty_cache()

        self.loss_train = loss_train
        self.acc_train = acc_train
        self.loss_test = loss_test
        self.acc_test = acc_test

    # pretrain client
    def pre_train_client(self):
        if len(self.data_train) == 0:
            return
        client_trainer = LocalUpdate(self.client_id, self.proj.lr_pretrain, self.proj.device,
                                     self.proj.BATCH_SIZE, self.data_train, self.data_test)
        client_trainer.local_ep = self.proj.epochs_client_local
        self.model.to(self.proj.device)
        w, _, _ = client_trainer.train(self.model)
        net = self.model
        net.load_state_dict(w)
        self.model = net
        self.model.to('cpu')
        torch.cuda.empty_cache()

    # Per weights training
    def train_weights(self):
        shard_weights = {shard_connection.shard_id: shard_connection.weight for shard_connection in
                         self.shard_connections}
        shard_models = {shard_connection.shard_id: shard_connection.model for shard_connection in
                        self.shard_connections}
        weights_trainer = Shard_Weights_trainer(self.proj, shard_models, shard_weights, self.shard)
        shard_weights = weights_trainer.train_shard_weights(self.data_train)
        for shard_id, weight in shard_weights.items():
            for shard_connection in self.shard_connections:
                if shard_connection.shard_id == shard_id:
                    shard_connection.weight = weight
        for sc in self.shard_connections:
            sc.model = None
            sc.weight.to('cpu')
        torch.cuda.empty_cache()

    def eval_client_data(self, net=None):
        if net is None:
            net = self.model
        client_trainer = LocalUpdate(self.client_id, self.proj.lr, self.proj.device,
                                     self.proj.BATCH_SIZE, self.data_train, self.data_test)
        loss_test, acc_test = client_trainer.evaluate(net.to(self.proj.device))

        return loss_test, acc_test

    # evaluate with Per
    def eval_client_shard(self, shard_connections=None):
        if shard_connections is None:
            shard_connections = self.shard_connections
        client_trainer = LocalUpdate(self.client_id, self.proj.lr_pretrain, self.proj.device,
                                     self.proj.BATCH_SIZE, self.data_train, self.data_test)
        accuracy = client_trainer.evaluate_connection(shard_connections, self.proj.num_classes)

        return accuracy

    def receive_model(self, glob_model):
        net = copy.deepcopy(self.proj.net_glob)
        net.load_state_dict(glob_model)
        self.model = net

    # receive shard models
    def receive_shard_connection(self, shard_connections):
        if not self.shard_connections:
            self.shard_connections = []
            for sc in shard_connections:
                new_sc = Shard_connection(sc.shard_id, copy.deepcopy(sc.weight))
                new_sc.model = sc.model
                self.shard_connections.append(new_sc)
        else:
            for sc_self in self.shard_connections:
                for sc in shard_connections:
                    if sc_self.shard_id == sc.shard_id:
                        sc_self.weight = copy.copy(sc.weight)
                        sc_self.model = sc.model

    # receive Per weights
    def receive_shard_connection_weight(self, shard_connections):
        for shard_connection in shard_connections:
            for shard_connection_self in self.shard_connections:
                if shard_connection_self.shard_id == shard_connection.shard_id:
                    shard_connection_self.weight = shard_connection.weight.clone()
                    break

    def upload_model(self):
        model_state = self.model.state_dict()
        loss_train = self.loss_train
        loss_test = self.loss_test
        acc_train = self.acc_train
        acc_test = self.acc_test

        return model_state, loss_train, loss_test, acc_train, acc_test

    # upload Per weights
    def upload_shard_connection_weight(self):
        return self.shard_connections

    def get_train_dataset_size(self):
        return len(self.data_train)

    def get_test_dataset_size(self):
        return len(self.data_test)

    def get_train_dataset_labels(self):
        return [self.data_train[i][1] for i in range(len(self.data_train))]
