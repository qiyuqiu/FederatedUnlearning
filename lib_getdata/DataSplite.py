from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from collections import defaultdict, Counter
import numpy as np
import torch
from torch.distributions.dirichlet import Dirichlet


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.targets = [label for _, label in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features, labels = self.data[index]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def dataset_iid(dataset, num_users):
    num_items = len(dataset) // num_users
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def dataset_nonidd_2(dataset, num_class, num_users, num_classes_per_user=2):
    dict_users = {i: [] for i in range(num_users)}
    idxs_labels = {i: [] for i in range(num_class)}

    for idx, (_, label) in enumerate(dataset):
        idxs_labels[label].append(idx)

    total_samples_per_client = len(dataset) // num_users - 2
    used_indices = set()

    for user in range(num_users):
        start_class = (user * num_classes_per_user) % num_class
        chosen_classes = [(start_class + i) % num_class for i in range(num_classes_per_user)]

        samples_per_class = total_samples_per_client // num_classes_per_user
        for cls in chosen_classes:
            available_samples = list(set(idxs_labels[cls]) - used_indices)
            if len(available_samples) < samples_per_class:
                samples = np.random.choice(available_samples, len(available_samples), replace=False)
            else:
                samples = np.random.choice(available_samples, samples_per_class, replace=False)
            dict_users[user].extend(samples)
            used_indices.update(samples)

    for user, data_indices in dict_users.items():
        user_labels = [dataset[idx][1] for idx in data_indices]
        class_counter = Counter(user_labels)
        print(f"Client {user}: {class_counter}")

    return dict_users


def dataset_non_iid_sigma(dataset, num_clients, sigma):
    num_samples = len(dataset)
    num_labels = len(np.unique([label for _, label in dataset]))
    n = num_samples // num_clients
    sigma_n = int(np.ceil(sigma * n))

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    label_based_data = defaultdict(list)
    for index in indices:
        _, label = dataset[index]
        label_based_data[label].append(index)

    clients_data = defaultdict(list)
    client_label_counts = defaultdict(lambda: defaultdict(int))

    for client_id in range(num_clients):
        main_label = np.random.choice(num_labels)
        main_label_indices = np.random.choice(label_based_data[main_label],
                                              min(sigma_n, len(label_based_data[main_label])), replace=False)
        clients_data[client_id].extend(main_label_indices)
        client_label_counts[client_id][main_label] += len(main_label_indices)
        label_based_data[main_label] = [i for i in label_based_data[main_label] if i not in main_label_indices]

    remaining_indices = [index for indices in label_based_data.values() for index in indices]
    np.random.shuffle(remaining_indices)
    for idx in remaining_indices:
        _, label = dataset[idx]
        for client_id in range(num_clients):
            if len(clients_data[client_id]) < n:
                clients_data[client_id].append(idx)
                client_label_counts[client_id][label] += 1
                break

    for client_id, label_counts in client_label_counts.items():
        print(f'Client {client_id} label counts: {label_counts}')

    return clients_data


# pathological Non-IID
def dataset_pathological_non_iid(dataset, num_users, num_classes, shards_per_user=5):
    dict_users = {}
    all_idxs = np.arange(len(dataset))
    labels = np.array([y for _, y in dataset])

    for user in range(num_users):
        dict_users[user] = set()
        classes_for_user = np.random.choice(range(num_classes), shards_per_user, replace=False)
        for cls in classes_for_user:
            idxs = all_idxs[labels == cls]
            dict_users[user].update(np.random.choice(idxs, len(idxs) // shards_per_user, replace=False))
            all_idxs = np.delete(all_idxs, np.where(labels == cls))

    return dict_users


# Dirichlet Non-IID
def dataset_dirichlet_non_iid(dataset, num_users, num_classes, alpha=0.5):
    def dirichlet_split_noniid(train_labels, alpha, n_clients, n_classes):
        label_distribution = Dirichlet(torch.full((n_clients,), alpha)).sample((n_classes,))
        class_idcs = [torch.nonzero(train_labels == y).flatten() for y in range(n_classes)]
        client_idcs = [[] for _ in range(n_clients)]

        for c, fracs in zip(class_idcs, label_distribution):
            total_size = len(c)
            splits = (fracs * total_size).int()
            splits[-1] = total_size - splits[:-1].sum()
            idcs = torch.split(c, splits.tolist())
            for i, idx in enumerate(idcs):
                client_idcs[i] += [idx]

        client_idcs = [torch.cat(idcs) for idcs in client_idcs]
        return client_idcs

    dict_users = {}
    labels = np.array([y for _, y in dataset])
    client_idcs = dirichlet_split_noniid(torch.tensor(labels), alpha, num_users, num_classes)

    for user, idcs in enumerate(client_idcs):
        dict_users[user] = set(idcs.numpy())

    return dict_users
