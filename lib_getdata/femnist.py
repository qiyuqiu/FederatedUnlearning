import json
import torch
from lib_client.Client_local import Client
import random
from typing import Dict, List, Tuple, Any
import os
import pickle
from lib_model.FEMNIST.CNN import *
from collections import Counter
import math

from lib_utils.untils import prRed


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def split_data(features: List[Any], labels: List[Any], test_ratio: float = 0.2) -> Tuple[List[Tuple], List[Tuple]]:
    combined = list(zip(features, labels))
    random.shuffle(combined)
    split_idx = int(len(combined) * (1 - test_ratio))
    train_data = combined[:split_idx]
    test_data = combined[split_idx:]
    return train_data, test_data


def create_clients(data: Dict[str, Any], proj) -> List[Client]:
    clients = []
    user_data = data['user_data']
    for user_id, content in user_data.items():
        features = torch.tensor(content['x'], dtype=torch.float)
        features = features.view(-1, 1, 28, 28)
        labels = torch.tensor(content['y'], dtype=torch.long)
        train_data, test_data = split_data(features, labels)
        clients.append(Client(user_id, train_data, test_data, project=proj))
    return clients


def process_directory(directory, proj):
    all_clients = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            data = load_data(file_path)
            clients = create_clients(data, proj)
            all_clients.extend(clients)
    return all_clients


def save_clients(clients, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(clients, f)


def load_clients(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def calculate_entropy(labels: torch.Tensor) -> float:
    label_count = Counter(labels.numpy())
    total = len(labels)
    entropy = 0.0
    for count in label_count.values():
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy


def find_most_uneven_clients(clients: List[Client], top_n: int = 500) -> List[Client]:
    client_entropy = []
    for client in clients:
        labels = []
        for _, label in client.data_train:
            if isinstance(label, torch.Tensor):
                if label.dim() == 0:
                    label = label.unsqueeze(0)
                if label.numel() > 0:
                    labels.append(label)
                else:
                    print(f"Skipping empty label for client {client.client_id}: {label}")
            else:
                print(f"Skipping invalid label for client {client.client_id}: {label}")

        if labels:
            try:
                labels_tensor = torch.cat(labels)
                entropy = calculate_entropy(labels_tensor)
                client_entropy.append((client, entropy))
            except RuntimeError as e:
                print(f"Error concatenating labels for client {client.client_id}: {e}")

    client_entropy.sort(key=lambda x: x[1])
    selected_clients = [client for client, _ in client_entropy[:top_n]]

    del client_entropy
    del labels
    del labels_tensor

    return selected_clients


def reassign_client_ids(clients):
    for new_id, client in enumerate(clients):
        client.client_id = new_id


def get_femnist_FL(proj):
    proj.num_classes = 62
    proj.net_glob = CNN(proj.num_classes)
    proj.plot_title = 'femnist, CNN.png'
    proj.num_client = 200
    proj.num_shards = 20
    proj.epochs = 300
    proj.lr = 0.001
    proj.epochs_weighted_train_interval = 10

    print("Data:FEMNIST")

    total_params = 0
    for param in proj.net_glob.parameters():
        total_params += param.numel()

    # Path to the FEMNIST dataset directory (update according to your environment)
    directory_path = 'data/FEMNIST/all_data'

    clients = process_directory(directory_path, proj)

    client_id = 0
    for client in clients:
        client.client_id = client_id
        client_id += 1

    clients = find_most_uneven_clients(clients, top_n=proj.num_client)
    proj.num_client = len(clients)
    reassign_client_ids(clients)
    for client in clients:
        print(client.client_id)
    del_clients = [client for client in clients if client not in clients[:proj.num_client]]
    del del_clients

    return clients
