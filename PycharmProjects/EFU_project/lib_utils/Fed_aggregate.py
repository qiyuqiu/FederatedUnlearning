import copy
import torch


# Federated averaging aggregation for model parameters
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# Weighted federated averaging aggregation for model parameters
def weighted_FedAvg(w, sample_counts):
    total_samples = sum(sample_counts)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] *= sample_counts[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * sample_counts[i]
        w_avg[k] = torch.div(w_avg[k], total_samples)
    return w_avg


# Aggregation of shard-level weight parameters
def weight_weighted_FedAvg(shard_weights_list, sample_counts):
    total_samples = sum(sample_counts)
    if total_samples == 0:
        raise ValueError("Total number of samples is zero.")

    # Initialize w_avg using the same key structure as shard_weights_list
    w_avg = {key: torch.clone(value) * sample_counts[0] for key, value in shard_weights_list[0].items()}

    # Iterate through each shard's weight dictionary and perform weighted accumulation
    for i in range(1, len(shard_weights_list)):
        for key in w_avg.keys():
            w_avg[key] += shard_weights_list[i][key] * sample_counts[i]

    # Normalize the aggregated weights
    for key in w_avg.keys():
        w_avg[key] /= total_samples

    return w_avg
