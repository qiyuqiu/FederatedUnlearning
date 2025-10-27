import copy
import random
import torch
from lib_utils.Fed_aggregate import weighted_FedAvg, FedAvg
from lib_server.Shard import Shard
from lib_utils.untils import *


def update_client_shard_id(clients, shard_id):
    for client in clients:
        client.shard = shard_id


# Dyn_1
def Adjust_group_numUnchanged_balance_initial(shards, proj):
    threshold_high = proj.num_client / proj.num_shards * 2
    threshold_low = threshold_high / 4

    changed_shards = []

    while True:
        exceed_threshold = [shard for shard in shards if len(shard.clients) >= threshold_high]

        if not exceed_threshold:
            break

        for shard in exceed_threshold:
            new_shard1, new_shard2 = random_split(shard.clients)
            # 获取每个客户端的样本数
            sample_counts1 = [len(client.data_train) for client in new_shard1]
            sample_counts2 = [len(client.data_train) for client in new_shard2]
            shard.clients = new_shard1
            current_shard_model_dict = weighted_FedAvg([client.model.state_dict()
                                                        for client in shard.clients], sample_counts1)
            shard.model.load_state_dict(current_shard_model_dict)
            update_client_shard_id(shard.clients, shard.shard_id)
            changed_shards.append(shard)
            shard_to_merge = min((s for s in shards), key=lambda s: len(s.clients))
            nearest_shard_idx = find_nearest_shard([shard.model.state_dict() for shard in shards],
                                                   shard_to_merge.shard_id)
            nearest_shard = shards[nearest_shard_idx]
            sample_counts_merge = [len(client.data_train) for client in shard_to_merge.clients + nearest_shard.clients]
            nearest_shard.clients.extend(shard_to_merge.clients)
            nearest_shard_model_dict = weighted_FedAvg([client.model.state_dict()
                                                        for client in nearest_shard.clients], sample_counts_merge)
            nearest_shard.model.load_state_dict(nearest_shard_model_dict)
            update_client_shard_id(nearest_shard.clients, nearest_shard.shard_id)
            changed_shards.append(nearest_shard)
            sample_counts_new_shard2 = [len(client.data_train) for client in new_shard2]
            shard_to_merge.clients = new_shard2
            shard_to_merge_model_dict = weighted_FedAvg([client.model.state_dict()
                                                         for client in new_shard2], sample_counts_new_shard2)
            shard_to_merge.model.load_state_dict(shard_to_merge_model_dict)
            update_client_shard_id(shard_to_merge.clients, shard_to_merge.shard_id)
            changed_shards.append(shard_to_merge)

    for shard in shards:
        prGreen(f"Shard {shard.shard_id} contains clients: {[client.client_id for client in shard.clients]}")

    return changed_shards


# Dyn_2
def Adjust_group_numUnchanged_strengthen_balance_initial(shards, proj):
    threshold_high = proj.num_client / proj.num_shards * 2
    threshold_low = threshold_high / 4
    changed_shards = []
    changed_shards_retrain = []

    while True:
        exceed_threshold = [shard for shard in shards if len(shard.clients) >= threshold_high]
        if not exceed_threshold:
            break
        for shard in exceed_threshold:
            new_shard1, new_shard2 = random_split(shard.clients)
            sample_counts1 = [len(client.data_train) for client in new_shard1]
            sample_counts2 = [len(client.data_train) for client in new_shard2]
            shard.clients = new_shard1
            current_shard_model_dict = weighted_FedAvg([client.model.state_dict()
                                                        for client in shard.clients], sample_counts1)
            shard.model.load_state_dict(current_shard_model_dict)
            update_client_shard_id(shard.clients, shard.shard_id)
            changed_shards.append(shard)
            changed_shards_retrain.append(shard)
            shard_to_merge = min((s for s in shards), key=lambda s: len(s.clients))
            nearest_shard_idx = find_nearest_shard([shard.model.state_dict() for shard in shards],
                                                   shard_to_merge.shard_id)
            nearest_shard = shards[nearest_shard_idx]

            sample_counts_merge = [len(client.data_train) for client in shard_to_merge.clients + nearest_shard.clients]
            nearest_shard.clients.extend(shard_to_merge.clients)
            nearest_shard_model_dict = weighted_FedAvg([client.model.state_dict()
                                                        for client in nearest_shard.clients], sample_counts_merge)
            nearest_shard.model.load_state_dict(nearest_shard_model_dict)
            update_client_shard_id(nearest_shard.clients, nearest_shard.shard_id)
            changed_shards.append(nearest_shard)
            sample_counts_new_shard2 = [len(client.data_train) for client in new_shard2]
            shard_to_merge.clients = new_shard2
            shard_to_merge_model_dict = weighted_FedAvg([client.model.state_dict()
                                                         for client in new_shard2], sample_counts_new_shard2)
            shard_to_merge.model.load_state_dict(shard_to_merge_model_dict)
            update_client_shard_id(shard_to_merge.clients, shard_to_merge.shard_id)
            changed_shards.append(shard_to_merge)
            changed_shards_retrain.append(shard_to_merge)

    if threshold_low <= 3:
        return changed_shards, changed_shards_retrain

    while True:
        below_threshold = [shard for shard in shards if len(shard.clients) < threshold_low]

        if not below_threshold:
            break

        for shard in below_threshold:
            if len(shard.clients) != 0:
                nearest_shard_idx = find_nearest_shard([shard.model.state_dict() for shard in shards], shard.shard_id)
                nearest_shard = shards[nearest_shard_idx]
                nearest_shard.clients.extend(shard.clients)
                sample_counts_merge = [len(client.data_train) for client in nearest_shard.clients]
                nearest_shard_model_dict = weighted_FedAvg(
                    [client.model.state_dict() for client in nearest_shard.clients],
                    sample_counts_merge)
                nearest_shard.model.load_state_dict(nearest_shard_model_dict)
                update_client_shard_id(nearest_shard.clients, nearest_shard.shard_id)
                changed_shards.append(nearest_shard)
            shard_to_split = max((s for s in shards if s.shard_id != 0), key=lambda s: len(s.clients))
            new_shard1, new_shard2 = random_split(shard_to_split.clients)
            sample_counts1 = [len(client.data_train) for client in new_shard1]
            sample_counts2 = [len(client.data_train) for client in new_shard2]
            shard_to_split.clients = new_shard1
            shard_to_split_model_dict = weighted_FedAvg([client.model.state_dict() for client in new_shard1],
                                                        sample_counts1)
            shard_to_split.model.load_state_dict(shard_to_split_model_dict)
            update_client_shard_id(shard_to_split.clients, shard_to_split.shard_id)
            changed_shards.append(shard_to_split)
            changed_shards_retrain.append(shard_to_split)
            shard.clients = new_shard2
            shard_model_dict = weighted_FedAvg([client.model.state_dict() for client in new_shard2], sample_counts2)
            shard.model.load_state_dict(shard_model_dict)
            update_client_shard_id(shard.clients, shard.shard_id)
            changed_shards.append(shard)
            changed_shards_retrain.append(shard)

    for shard in shards:
        prGreen(f"Shard {shard.shard_id} contains clients: {[client.client_id for client in shard.clients]}")

    return changed_shards, changed_shards_retrain


def random_split(current_clients):
    shard1 = random.sample(current_clients, len(current_clients) // 2)
    shard2 = list(set(current_clients) - set(shard1))
    return shard1, shard2


def find_nearest_shard(shard_models, current_shard_index):
    current_shard_index -= 1
    current_model = shard_models[current_shard_index - 1]
    current_model_tensors = [value.clone().detach().to(dtype=torch.float32).view(-1) for value in
                             current_model.values()]
    current_model_tensor = torch.cat(current_model_tensors)
    min_distance = float('inf')
    nearest_shard_index = None
    for i, shard_weight in enumerate(shard_models):
        if i == current_shard_index:
            continue
        shard_weight_tensors = [value.clone().detach().to(dtype=torch.float32).view(-1) for value in
                                shard_weight.values()]
        shard_weight_tensor = torch.cat(shard_weight_tensors)
        distance = torch.norm(current_model_tensor - shard_weight_tensor).item()
        if distance < min_distance:
            min_distance = distance
            nearest_shard_index = i

    return nearest_shard_index
