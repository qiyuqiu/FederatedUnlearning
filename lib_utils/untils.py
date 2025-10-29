import random
import time
import numpy as np
from attribute_FL import proj
import torch
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt

proj = proj()


def is_model_or_state_dict(obj):
    """
    Determine whether the input object is a model or a state dictionary.

    Args:
    - obj: The object to check.

    Returns:
    - 'model': If the object is a model instance.
    - 'state_dict': If the object is a state dictionary.
    """
    if isinstance(obj, nn.Module):
        return 'model'
    elif isinstance(obj, OrderedDict):
        return 'state_dict'
    else:
        raise ValueError("The object is neither a model nor a state_dict.")


def prRed(skk): print("\033[91m {}\033[00m".format(skk))


def prGreen(skk): print("\033[92m {}\033[00m".format(skk))


def move_state_dict_to_device(state_dict, device):
    return {key: value.to(device) for key, value in state_dict.items()}


def move_state_dict_to_cpu(state_dict):
    return {key: value.to('cpu') for key, value in state_dict.items()}


def print_device_info(name, state_dict):
    for key, value in state_dict.items():
        print(f"{name} - {key} is on device: {value.device}")


def findShard(shards, shard_id):
    for shard in shards:
        if shard.shard_id == shard_id:
            return shard
    print('Shard not found, returning shard with shard_id == 0')
    return findShard(shards, 0)


def split_data_into_chunks(data_train, num_chunk):
    # Assume data_train is a list or array containing all training data
    total_data = len(data_train)
    num_splits = num_chunk  # Number of chunks to split into
    chunk_size = total_data // num_splits  # Size of each chunk

    # Generate all indices
    all_indices = list(range(total_data))

    # Shuffle indices
    random.shuffle(all_indices)

    # Split indices into chunks
    index_chunks = [all_indices[i * chunk_size:(i + 1) * chunk_size] for i in range(num_splits)]

    # Distribute any remaining indices due to non-divisible total size
    if total_data % num_splits != 0:
        remaining_indices = all_indices[num_splits * chunk_size:]
        for i, idx in enumerate(remaining_indices):
            index_chunks[i].append(idx)

    # Return the chunked indices
    return index_chunks


# Compare two models and check if their parameters are identical
def are_models_equal(model1, model2):
    model1_dict = model1.state_dict()
    model2_dict = model2.state_dict()

    for key_item1, key_item2 in zip(model1_dict.items(), model2_dict.items()):
        key1, item1 = key_item1
        key2, item2 = key_item2

        if key1 != key2 or not torch.equal(item1, item2):
            return False

    return True


# Simulate communication delay
def simulate_communication_delay(data_size_kb, time_per_unit=proj.time_per_unit):
    """
    Simulate communication delay.
    Args:
    - data_size_kb: Size of data to be transmitted in kilobytes.
    - time_per_unit: Time taken per unit of data in seconds.
    """
    delay = data_size_kb * time_per_unit
    time.sleep(delay)


# Check whether the training data contains label 0
def contains_label_0(data_train):
    for data, label in data_train:
        if label == 0:
            return True
    return False


# Filter out samples with label 0 from the test dataset
def filter_label_0(data_test):
    data_test_label0 = []
    for data, label in data_test:
        if label == 0:
            data_test_label0.append((data, label))
    return data_test_label0


def plot_accuracy_curve(data, title):
    """
    Plot accuracy curve over epochs.

    Args:
    data (list): A list of accuracy values.
    title (str): Title of the plot.

    Returns:
    None
    """
    epochs = list(range(1, len(data) + 1))  # Generate list of epoch numbers

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data, marker='o', linestyle='-', color='b', label='Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
