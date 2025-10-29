import copy
import gc
import os
import random

import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
from lib_cluster.model_Kmeans import model_Kmeans
from lib_cluster.model_hierarchical import model_Hierarchical
from lib_cluster.random_avg import random_shard_assignment_avg
from lib_rank.rank import Rank
from lib_server.Shard import Shard
from lib_DynamicGroup.DynamicGroup import *
from lib_utils.untils import *
from lib_utils.Fed_aggregate import weighted_FedAvg, FedAvg
from lib_connection_train.Shard_connection import Shard_connection
import multiprocessing as mp


class server:
    def __init__(self, proj, clients):
        self.early_stopping_rounds = proj.early_stopping_rounds
        self.clients = clients
        self.model = copy.deepcopy(proj.net_glob)
        self.lr = proj.lr
        self.epochs = proj.epochs
        self.shards = []
        self.proj = proj

        # Initial connections between each shard and others
        self.shard_connection_initial = []

        # Storage for all ranks
        self.ranks = []

        # Counter for the number of dynamic adjustments
        self.Adjusted_epoch = 0

        # Lists to store data for each round
        self.loss_train_epoch = []
        self.loss_test_epoch = []
        self.acc_train_epoch = []
        self.acc_test_epoch = []

        # To store the accuracy for each round with and without using connections
        self.acc_shard_test_epoch = []

        # To store the accuracy curve for the FedCIO method
        self.acc_FedCIO_epoch = []

        # To store accuracy results from hierarchical training
        self.acc_rank_training_epoch = []

        # To store overall accuracy results from SISA training
        self.acc_SISA = []

    # =============================================================================
    #                                   FLB
    # =============================================================================
    def train_FL(self):
        max_test_acc = 0
        consecutive_no_improvement = 0
        for epoch in range(self.epochs):
            self.broadcast()
            num_clients_to_sample = int(len(self.clients) * self.proj.client_fraction)
            sampled_clients = random.sample(self.clients, num_clients_to_sample)
            with tqdm(total=len(sampled_clients), desc="Clients Training") as pbar:
                for client in sampled_clients:
                    if client.iswork == 0:
                        continue
                    client.local_update()
                    pbar.update(1)
            self.aggregate_Model(sampled_clients)
            print(f'Epoch {epoch}')
            loss, acc = self.eval_model_epoch()
            print()
            if self.proj.tag_is_stop_server:
                if acc <= max_test_acc:
                    consecutive_no_improvement += 1
                else:
                    max_test_acc = acc
                    consecutive_no_improvement = 0
                if consecutive_no_improvement >= self.early_stopping_rounds:
                    prRed(f'No improvement in test accuracy for {self.early_stopping_rounds}'
                          f' consecutive rounds. Stopping early.')
                    break
        self.plot_training_curves()
        print("FL training:")
        print(self.acc_test_epoch)
        return self.model

    # Federated Learning unlearning and retraining
    def FL_unlearn(self, client):
        self.proj.num_client -= 1
        client_unlearn = None
        for c in self.clients:
            if c.client_id == client.client_id:
                client_unlearn = c
                break
        if client_unlearn in self.clients:
            self.clients.remove(client_unlearn)
        else:
            print(f"Client {client.client_id} not found in clients list.")
            return
        self.model = copy.deepcopy(self.proj.net_glob)
        self.train_FL()

    # =============================================================================
    #                                   SISA
    # =============================================================================
    def train_FL_SISA(self):
        # Initialize all shards
        for shard_id in range(self.proj.num_shards + 1):
            shard = Shard(shard_id, self.proj)
            shard.model = copy.deepcopy(self.model)
            self.shards.append(shard)
        shard_0 = findShard(self.shards, 0)
        self.shards.remove(shard_0)

        # Distribute clients to shards based on unlearn probability
        sorted(self.clients, key=lambda client: client.unlearn_prob)
        num_clients_per_shard = int(len(self.clients) / len(self.shards))
        client_index = 0
        for shard in self.shards:
            shard.clients = self.clients[client_index: client_index + num_clients_per_shard]
            client_index += num_clients_per_shard
            for client in self.clients:
                client.shard = shard.shard_id

        # Start training using SFL approach
        max_test_acc = 0
        consecutive_no_improvement = 0
        tag_epochs_weight_train = 0
        for epoch in range(self.epochs):
            self.broadcast_shard_model()
            num_clients_to_sample = int(len(self.clients) * self.proj.client_fraction)
            sampled_clients = random.sample(self.clients, num_clients_to_sample)
            with tqdm(total=len(sampled_clients), desc="Clients Training") as pbar:
                for client in sampled_clients:
                    if client.is_trained_completely:
                        continue
                    client.local_update()
                    pbar.update(1)
            self.aggregate_shardModel(sampled_clients)
            print(f'Epoch {epoch + 1}')
            acc = self.aggregate_shard_models_test()
            self.acc_SISA.append(acc)
            if self.proj.tag_is_stop_server:
                if acc <= max_test_acc:
                    consecutive_no_improvement += 1
                else:
                    max_test_acc = acc
                    consecutive_no_improvement = 0
                if consecutive_no_improvement >= self.early_stopping_rounds:
                    prRed(
                        f'No improvement in test accuracy for {self.early_stopping_rounds} consecutive rounds. Stopping early.')
                    break
        for shard in self.shards:
            shard.is_trained_completely = True
            for client in shard.clients:
                client.is_trained_completely = True
        self.plot_accuracy_curves()
        print("SISA_FL training:")
        print(self.acc_FedCIO_epoch)
        return self.shards

    def train_FL_SISA_unlearn(self, clients_unlearn):
        self.Adjusted_epoch += len(clients_unlearn)
        self.proj.num_client -= len(clients_unlearn)
        shards_retrain = []
        for current_client in clients_unlearn:
            current_shard = next(
                shard for shard in self.shards if shard.shard_id != 0 and current_client in shard.clients)
            if current_client in self.clients:
                self.clients.remove(current_client)
            else:
                print(f"Client {current_client.client_id} not found in clients list.")
                return
            if current_client in current_shard.clients:
                current_shard.clients.remove(current_client)
            print(f'Client {current_client.client_id} unlearned from shard {current_shard.shard_id}...')
            if current_shard not in shards_retrain:
                shards_retrain.append(current_shard)
            current_shard.model = None
            del current_shard.model
            gc.collect()
            current_shard.model = copy.deepcopy(self.proj.net_glob)
        for shard in shards_retrain:
            shard.is_trained_completely = False
            for client in shard.clients:
                client.is_trained_completely = False
        max_test_acc = 0
        consecutive_no_improvement = 0
        clients_in_shard_retrain = []
        for shard in shards_retrain:
            for client in shard.clients:
                clients_in_shard_retrain.append(client)
        for epoch in range(self.proj.epochs):
            num_clients_to_sample = int(len(clients_in_shard_retrain) * self.proj.client_fraction)
            sampled_clients = random.sample(clients_in_shard_retrain, num_clients_to_sample)
            print("Sampled Clients in this round:")
            for client in sampled_clients:
                print(f"Client ID: {client.client_id}, Shard ID: {client.shard}")
            self.broadcast_shard_model()
            for client in sampled_clients:
                client.local_update()
            self.aggregate_shardModel(sampled_clients)
            print(f'SISA-clientsUnlearning: Epoch {epoch + 1}')
            acc = self.aggregate_shard_models_test()
            print()
            self.acc_SISA.append(acc)
            if self.proj.tag_is_stop_server:
                if acc <= max_test_acc:
                    consecutive_no_improvement += 1
                else:
                    max_test_acc = acc
                    consecutive_no_improvement = 0
                if consecutive_no_improvement >= self.early_stopping_rounds:
                    prRed(
                        f'No improvement in test accuracy for {self.early_stopping_rounds} consecutive rounds. Stopping early.')
                    break
        for shard in self.shards:
            shard.is_trained_completely = True
            for client in shard.clients:
                client.is_trained_completely = True
        self.plot_training_curves()
        self.plot_accuracy_curves()
        print("SISA-training and unlearning:")
        print(self.acc_SISA)
        print('Shard accuracy for retrained shards:')
        for shard in shards_retrain:
            print(f'Shard {shard.shard_id} accuracy:')
            print(shard.acc_test_epoch)
        print("Clients successfully unlearned.")

    # =============================================================================
    #                       EFU-Per,FedCIO
    # =============================================================================
    # our paper
    def train_SFL(self):
        for shard_id in range(self.proj.num_shards + 1):
            shard = Shard(shard_id, self.proj)
            shard.model = copy.deepcopy(self.model)
            self.shards.append(shard)
        shard_0 = findShard(self.shards, 0)
        self.shards.remove(shard_0)

        # initial group clients
        self.broadcast()
        with tqdm(total=len(self.clients), desc="Clients preTraining") as pbar:
            for client in self.clients:
                client.pre_train_client()
                pbar.update(1)
        if self.proj.cluster_strategy == 1:
            self.receive_model_km()
        elif self.proj.cluster_strategy == 2:
            self.receive_model_hierarchical()
        elif self.proj.cluster_strategy == 3:
            self.receive_model_km()
            Adjust_group_numUnchanged_balance_initial(self.shards, self.proj)
        elif self.proj.cluster_strategy == 4:
            self.receive_model_km()
            Adjust_group_numUnchanged_strengthen_balance_initial(self.shards, self.proj)
        shard_connections = []
        for shard in self.shards:
            weight = torch.nn.Parameter(torch.tensor(1.0))
            new_shard_connection = Shard_connection(shard.shard_id, weight)
            shard_connections.append(new_shard_connection)
        self.shard_connection_initial = shard_connections
        for sc in self.shard_connection_initial:
            for shard in self.shards:
                if sc.shard_id == shard.shard_id:
                    sc.model = shard.model
                    break
        for shard in self.shards:
            shard.shard_connections = copy.deepcopy(self.shard_connection_initial)

        # shard training
        max_test_acc = 0
        consecutive_no_improvement = 0
        tag_epochs_weight_train = 0
        for epoch in range(self.epochs):
            self.broadcast_shard_model()
            num_clients_to_sample = int(len(self.clients) * self.proj.client_fraction)
            sampled_clients = random.sample(self.clients, num_clients_to_sample)
            with tqdm(total=len(sampled_clients), desc="Clients Training") as pbar:
                for client in sampled_clients:
                    if client.is_trained_completely:
                        continue
                    client.local_update()
                    pbar.update(1)
            self.aggregate_shardModel(sampled_clients)
            self.update_shards_connection()
            if epoch % self.proj.epochs_weighted_train_interval == 0:
                tag_epochs_weight_train = 0
                self.connection_train()
                for shard in self.shards:
                    print(f'shard {shard.shard_id} connections:')
                    for sc in shard.shard_connections:
                        print(f'shard {sc.shard_id}, weight: {sc.weight}')
                    print()
                    break
            print(f'Epoch {epoch + 1}')
            acc, acc_connection = self.eval_accuracy_epoch()
            self.aggregate_shard_models_test()
            print()
            if self.proj.tag_is_stop_server:
                if acc <= max_test_acc:
                    consecutive_no_improvement += 1
                else:
                    max_test_acc = acc
                    consecutive_no_improvement = 0
                if consecutive_no_improvement >= self.early_stopping_rounds:
                    prRed(f'No improvement in test accuracy for {self.early_stopping_rounds}'
                          f' consecutive rounds. Stopping early.')
                    break
        for shard in self.shards:
            shard.is_trained_completely = True
            for client in shard.clients:
                client.is_trained_completely = True
        self.plot_accuracy_curves()
        print("without connection training:")
        print(self.acc_test_epoch)
        print('connection training:')
        print(self.acc_shard_test_epoch)

        return self.shards

    # Per weights training phase
    def connection_train(self):
        with tqdm(total=len(self.shards), desc="shard connection Training") as pbar:
            for shard in self.shards:
                if shard.shard_id == 0:
                    continue
                for sc in shard.shard_connections:
                    for sc_initial in self.shard_connection_initial:
                        sc.weight = sc_initial.weight
                shard.train_weights_FL()
                pbar.update(1)

    # revoke a client
    def SFL_unlearn_client(self, client_unlearn):
        # delete the client
        self.Adjusted_epoch += 1
        self.proj.num_client -= 1
        shard_retrain = next(shard for shard in self.shards if shard.shard_id != 0 and client_unlearn in shard.clients)
        if client_unlearn in shard_retrain.clients:
            shard_retrain.clients.remove(client_unlearn)
        if client_unlearn in self.clients:
            self.clients.remove(client_unlearn)
        else:
            print(f"Client {client_unlearn.client_id} not found in clients list.")
            return
        print(f'client{client_unlearn.client_id} unlearn from shard{shard_retrain.shard_id}...')
        if self.proj.is_save_client_model and client_unlearn.model_before_train is None:
            print('client未参与训练，不需要unlearn')
            return
        if self.proj.is_save_client_model:
            shard_retrain.model = copy.deepcopy(client_unlearn.model_before_train)
        else:
            shard_retrain.model = copy.deepcopy(self.proj.net_glob)
        self.update_shards_connection()
        shard_retrain.is_trained_completely = False
        for client in shard_retrain.clients:
            client.is_trained_completely = False

        # retrain the shard
        max_test_acc = 0
        consecutive_no_improvement = 0
        for epoch in range(self.proj.epochs):
            shard_retrain.broadcast_model()
            for client in shard_retrain.clients:
                client.local_update()
            self.aggregate_Model(shard_retrain.clients, shard_retrain)
            self.update_shards_connection()
            print(f'SFL-Unlearning：Epoch {epoch + 1}')
            acc, acc_connection = self.eval_accuracy_epoch()
            print()
            if self.proj.tag_is_stop_server:
                if acc <= max_test_acc:
                    consecutive_no_improvement += 1
                else:
                    max_test_acc = acc
                    consecutive_no_improvement = 0
                if consecutive_no_improvement >= self.early_stopping_rounds:
                    prRed(f'No improvement in test accuracy for {self.early_stopping_rounds}'
                          f' consecutive rounds. Stopping early.')
                    break
            self.aggregate_shard_models_test()
        if self.Adjusted_epoch == self.proj.epoch_to_adjust:
            self.connection_train()
            self.Adjusted_epoch = 0
        for shard in self.shards:
            shard.is_trained_completely = True
            for client in shard.clients:
                client.is_trained_completely = True
        self.plot_training_curves()
        self.plot_accuracy_curves()
        print("without connection training and unlearn:")
        print(self.acc_test_epoch)
        print('connection training and unlearn:')
        print(self.acc_shard_test_epoch)
        print(f"Client {client_unlearn.client_id} successfully unlearned from shard {shard_retrain.shard_id}.")

    # update shard model
    def update_shards_connection(self):
        for shard in self.shards:
            for shard_connection in shard.shard_connections:
                shard_connection.model = findShard(self.shards, shard_connection.shard_id).model

    # =============================================================================
    #                               FL+HC
    # =============================================================================
    # FL+HC
    def train_SFL_initial_FL(self):
        # FL and HC phase
        for epoch in range(self.proj.epochs_pretrain_FL):
            self.broadcast()
            for client in self.clients:
                client.local_update()
            self.aggregate_Model(self.clients)
            _, acc_test_avg = self.eval_model_epoch()
            print(f'FL: Epoch - {epoch + 1}, accuracy: {acc_test_avg}')
        for shard_id in range(self.proj.num_shards + 1):
            shard = Shard(shard_id, self.proj)
            shard.model = copy.deepcopy(self.model)
            self.shards.append(shard)
        shard_0 = findShard(self.shards, 0)
        self.shards.remove(shard_0)
        self.broadcast()
        with tqdm(total=len(self.clients), desc="Clients preTraining") as pbar:
            for client in self.clients:
                client.pre_train_client()
                pbar.update(1)
        self.receive_model_hierarchical()

        # group training
        max_test_acc = 0
        consecutive_no_improvement = 0
        for epoch in range(self.epochs - self.proj.epochs_pretrain_FL):
            self.broadcast_shard_model()
            num_clients_to_sample = int(len(self.clients) * self.proj.client_fraction)
            sampled_clients = random.sample(self.clients, num_clients_to_sample)
            with tqdm(total=len(sampled_clients), desc="Clients Training") as pbar:
                for client in sampled_clients:
                    if client.is_trained_completely:
                        continue
                    client.local_update()
                    pbar.update(1)
            self.aggregate_shardModel(sampled_clients)
            print(f'HC: Epoch {epoch + 1}')
            acc = self.eval_shardModel()
            print()
            if self.proj.tag_is_stop_server:
                if acc <= max_test_acc:
                    consecutive_no_improvement += 1
                else:
                    max_test_acc = acc
                    consecutive_no_improvement = 0
                if consecutive_no_improvement >= self.early_stopping_rounds:
                    prRed(f'No improvement in test accuracy for {self.early_stopping_rounds}'
                          f' consecutive rounds. Stopping early.')
                    break
        self.plot_training_curves()
        print("FL + HC:")
        print(self.acc_test_epoch)

        return self.shards

    # retrain
    def FL_HC_unlearn(self, client):
        self.proj.num_client -= 1
        client_unlearn = None
        for c in self.clients:
            if c.client_id == client.client_id:
                client_unlearn = c
                break
        if client_unlearn in self.clients:
            self.clients.remove(client_unlearn)
        else:
            print(f"Client {client.client_id} not found in clients list.")
            return
        self.model = copy.deepcopy(self.proj.net_glob)
        self.train_SFL_initial_FL()

    # =============================================================================
    #                               FATS
    # =============================================================================
    def FATS(self):
        max_test_acc = 0
        consecutive_no_improvement = 0
        for epoch in range(self.epochs):
            self.broadcast()
            num_clients_to_sample = int(len(self.clients) * self.proj.client_fraction)
            sampled_clients = random.sample(self.clients, num_clients_to_sample)
            with tqdm(total=len(sampled_clients), desc="Clients Training") as pbar:
                for client in sampled_clients:
                    client.local_update()
                    pbar.update(1)
            self.aggregate_Model(sampled_clients)
            print(f'FATS: Epoch {epoch}')
            loss, acc = self.eval_model_epoch()
            print()
            if self.proj.tag_is_stop_server:
                if acc <= max_test_acc:
                    consecutive_no_improvement += 1
                else:
                    max_test_acc = acc
                    consecutive_no_improvement = 0
                if consecutive_no_improvement >= self.early_stopping_rounds:
                    prRed(f'No improvement in test accuracy for {self.early_stopping_rounds}'
                          f' consecutive rounds. Stopping early.')
                    break
        self.plot_training_curves()
        print("FATS training:")
        print(self.acc_test_epoch)

        return self.model

    def FATS_unlearn(self, client):
        # delete and rollback
        self.proj.num_client -= 1
        if client in self.clients:
            self.clients.remove(client)
        else:
            print(f"Client {client.client_id} not found in clients list.")
            return
        for x_client in self.clients:
            x_client.is_trained_completely = False
        if client.model_before_train is None:
            prRed(f'client {client.client_id}, not join')
            return
        self.model = copy.deepcopy(client.model_before_train)
        print(f'client{client.client_id} unlearning...')

        # retrain from checkpoint
        max_test_acc = 0
        consecutive_no_improvement = 0
        for epoch in range(self.epochs):
            self.broadcast()
            num_clients_to_sample = int(len(self.clients) * self.proj.client_fraction)
            sampled_clients = random.sample(self.clients, num_clients_to_sample)
            with tqdm(total=len(sampled_clients), desc="Clients Training") as pbar:
                for client in sampled_clients:
                    client.local_update()
                    pbar.update(1)
            self.aggregate_Model(sampled_clients)
            print(f'FATS Unlearning: Epoch {epoch}')
            loss, acc = self.eval_model_epoch()
            print()
            if self.proj.tag_is_stop_server:
                if acc <= max_test_acc:
                    consecutive_no_improvement += 1
                else:
                    max_test_acc = acc
                    consecutive_no_improvement = 0
                if consecutive_no_improvement >= self.early_stopping_rounds:
                    prRed(f'No improvement in test accuracy for {self.early_stopping_rounds}'
                          f' consecutive rounds. Stopping early.')
                    break
        self.plot_training_curves()
        print("FATS training with Unlearn:")
        print(self.acc_test_epoch)

    # =============================================================================
    #                              Broadcast
    # =============================================================================
    def broadcast(self):
        for client in self.clients:
            client.receive_model(self.model.state_dict())

    def broadcast_shard_model(self):
        for client in self.clients:
            for shard in self.shards:
                if shard.shard_id == client.shard:
                    client.receive_model(shard.model.state_dict())
                    break

    # =============================================================================
    #                    Receive and Aggregate model:FedAvg
    # =============================================================================
    # aggregate global model
    def aggregate_Model(self, sampled_clients, shard=None):
        model_list = []
        train_sample_counts = []
        clients_work = [client for client in self.clients if client in sampled_clients]
        for client in clients_work:
            model, _, _, _, _ = client.upload_model()
            train_sample_count = client.get_train_dataset_size()
            train_sample_counts.append(train_sample_count)
            model_list.append(model)
        w_avg = weighted_FedAvg(model_list, train_sample_counts)
        if shard:
            shard.model.load_state_dict(w_avg)
        else:
            self.model.load_state_dict(w_avg)

    # aggregate shard model
    def aggregate_shardModel(self, sampled_clients):
        for shard in self.shards:
            if shard.shard_id == 0 or shard.is_trained_completely:
                continue
            shard_model_list = []
            test_sample_counts, train_sample_counts = [], []
            shard_clients_work = [client for client in shard.clients if client in sampled_clients]
            if len(shard_clients_work) == 0:
                break
            for client in shard_clients_work:
                model, _, _, _, _ = client.upload_model()
                sample_train_count = client.get_train_dataset_size()
                train_sample_counts.append(sample_train_count)
                shard_model_list.append(model)
            w_avg = weighted_FedAvg(shard_model_list, train_sample_counts)
            shard.model.load_state_dict(w_avg)

    # =============================================================================
    #                              Evaluate
    # =============================================================================
    # evaluate model
    def eval_model_epoch(self):
        acc_model_list = []
        loss_model_list = []
        samples_test_client_list = []
        for client in self.clients:
            loss, acc = client.eval_client_data(self.model)
            samples_test_client = client.get_test_dataset_size()

            acc_model_list.append(acc)
            loss_model_list.append(loss)
            samples_test_client_list.append(samples_test_client)
        total_samples = sum(samples_test_client_list)
        weighted_acc_model = sum(
            acc * samples for acc, samples in zip(acc_model_list, samples_test_client_list)) / total_samples
        weighted_loss_model = sum(
            acc * samples for acc, samples in zip(loss_model_list, samples_test_client_list)) / total_samples

        self.acc_test_epoch.append(weighted_acc_model)
        self.loss_test_epoch.append(weighted_loss_model)

        print("------------------------------------------------")
        prRed(f'Test loss: {weighted_loss_model}')
        prGreen(f'Test acc: {weighted_acc_model}')
        print("------------------------------------------------")

        return weighted_loss_model, weighted_acc_model

    # evaluate model by Per
    def eval_shardModel(self):
        acc_model_list = []
        samples_test_client_list = []
        for shard in self.shards:
            for client in shard.clients:
                _, acc = client.eval_client_data(shard.model)
                samples_test_client = client.get_test_dataset_size()
                acc_model_list.append(acc)
                samples_test_client_list.append(samples_test_client)
            if self.proj.tag_is_stop_shard:
                shard.eval_is_stop_train()
        total_samples = sum(samples_test_client_list)
        weighted_acc_model = sum(
            acc * samples for acc, samples in zip(acc_model_list, samples_test_client_list)) / total_samples
        self.acc_test_epoch.append(weighted_acc_model)
        print("------------------------------------------------")
        prRed(f'Acc: {weighted_acc_model}')
        print("------------------------------------------------")

        return weighted_acc_model

    def eval_accuracy_epoch(self):
        acc_model_list = []
        acc_connection_list = []
        samples_test_client_list = []
        for shard in self.shards:
            shard_acc_list = []
            for client in shard.clients:
                _, acc = client.eval_client_data(shard.model)
                acc_connection = client.eval_client_shard(shard.shard_connections)
                samples_test_client = client.get_test_dataset_size()
                shard_acc_list.append(acc)
                acc_model_list.append(acc)
                acc_connection_list.append(acc_connection)
                samples_test_client_list.append(samples_test_client)
            if shard_acc_list:
                shard_acc = sum(shard_acc_list) / len(shard_acc_list)
            else:
                shard_acc = 0.0
            shard.acc_test_epoch.append(shard_acc)
        total_samples = sum(samples_test_client_list)
        weighted_acc_model = sum(
            acc * samples for acc, samples in zip(acc_model_list, samples_test_client_list)) / total_samples
        weighted_acc_connection = sum(
            acc * samples for acc, samples in zip(acc_connection_list, samples_test_client_list)) / total_samples
        self.acc_test_epoch.append(weighted_acc_model)
        self.acc_shard_test_epoch.append(weighted_acc_connection)
        print("------------------------------------------------")
        prRed(f'predict without connection: {weighted_acc_model}')
        prGreen(f'predict with connection: {weighted_acc_connection}')
        print("------------------------------------------------")

        return weighted_acc_model, weighted_acc_connection

    # aggregate model avg
    def aggregate_shard_models_test(self):
        device = torch.device(self.proj.device)
        state_dicts = [move_state_dict_to_device(shard.model.state_dict(), device) for shard in self.shards]
        model_dict = FedAvg(state_dicts)
        self.model.load_state_dict(model_dict)
        loss_list, acc_list = [], []
        test_sample_counts = []
        for client in self.clients:
            loss, acc = client.eval_client_data(self.model.to(self.proj.device))
            loss_list.append(loss)
            acc_list.append(acc)
            test_sample_count = len(client.data_test)
            test_sample_counts.append(test_sample_count)
        total_test_samples = sum(test_sample_counts)
        loss_test_avg = sum(
            loss * count for loss, count in zip(loss_list, test_sample_counts)) / total_test_samples
        acc_test_avg = sum(
            acc * count for acc, count in zip(acc_list, test_sample_counts)) / total_test_samples
        print('shard models avg')
        prGreen(f'loss: {loss_test_avg}, acc: {acc_test_avg}')
        self.acc_FedCIO_epoch.append(acc_test_avg)

        return acc_test_avg

    # =============================================================================
    #                              Client Cluster
    # =============================================================================
    # Kmeans
    def receive_model_km(self):
        client_models = []
        for client in self.clients:
            model, _, _, _, _ = client.upload_model()
            client_models.append(model)
        allocated = torch.cuda.memory_allocated()
        print(f"Current GPU memory allocated: {allocated / 1024 ** 3:.2f} GB")
        max_allocated = torch.cuda.max_memory_allocated()
        print(f"Max GPU memory allocated: {max_allocated / 1024 ** 3:.2f} GB")
        if self.proj.num_shards == len(self.clients):
            shard_clients_ids = random_shard_assignment_avg(len(self.clients), self.proj.num_shards)
        else:
            shard_clients_ids = model_Kmeans(self.proj, client_models)
        allocated = torch.cuda.memory_allocated()
        print(f"Current GPU memory allocated: {allocated / 1024 ** 3:.2f} GB")
        max_allocated = torch.cuda.max_memory_allocated()
        print(f"Max GPU memory allocated: {max_allocated / 1024 ** 3:.2f} GB")
        for shard in self.shards:
            if shard.shard_id == 0:
                continue
            current_shard_clients = []
            for client_id in shard_clients_ids[shard.shard_id - 1]:
                current_shard_clients.append(self.clients[client_id])
                self.clients[client_id].shard = shard.shard_id
            shard.clients = current_shard_clients
        for shard in self.shards:
            if shard.shard_id == 0:
                continue
            net_list = [client_models[client_id] for client_id in shard_clients_ids[shard.shard_id - 1]]
            # train_samples_counts_list = [client.get_train_dataset_size() for client in shard.clients]
            net_avg = FedAvg(net_list)
            shard.model.load_state_dict(net_avg)
        allocated = torch.cuda.memory_allocated()
        print(f"Current GPU memory allocated: {allocated / 1024 ** 3:.2f} GB")
        max_allocated = torch.cuda.max_memory_allocated()
        print(f"Max GPU memory allocated: {max_allocated / 1024 ** 3:.2f} GB")

    # HC
    def receive_model_hierarchical(self):
        client_models = []
        for client in self.clients:
            model, _, _, _, _ = client.upload_model()
            client_models.append(model)
        shard_clients_ids = model_Hierarchical(self.proj, client_models)
        for shard in self.shards:
            if shard.shard_id == 0:
                continue
            current_shard_clients = []
            for client_id in shard_clients_ids[shard.shard_id - 1]:
                current_shard_clients.append(self.clients[client_id])
                self.clients[client_id].shard = shard.shard_id
            shard.clients = current_shard_clients
        for shard in self.shards:
            if shard.shard_id == 0:
                continue
            net_list = [client.model.state_dict() for client in shard.clients]
            print(len(net_list))
            net_avg = FedAvg(net_list)
            shard.model.load_state_dict(net_avg)

    # =============================================================================
    #                              Results Plot
    # =============================================================================
    def plot_training_curves(self):
        epochs = range(1, len(self.acc_test_epoch) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.acc_test_epoch, 'ro-', label='Test Accuracy')
        plt.title('Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def plot_accuracy_curves(self):
        if len(self.acc_shard_test_epoch) != len(self.acc_test_epoch):
            raise ValueError("The lengths of acc_shard_test_epoch and acc_test_epoch must be the same.")
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.acc_shard_test_epoch) + 1)
        plt.plot(epochs, self.acc_shard_test_epoch, label='Shard Weighted Test Accuracy')
        plt.plot(epochs, self.acc_test_epoch, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison Over Epochs')
        plt.legend()
        plt.grid(True)
        save_dir = 'results of experiment'
        save_path = os.path.join(save_dir, self.proj.plot_title)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)

        plt.show()
