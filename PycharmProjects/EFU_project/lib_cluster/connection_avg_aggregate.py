from lib_connection_train.Shard_connection import Shard_connection


# aggregate personalized weights
def aggregate_connection(shard_connections_list):
    shard_weights = {}
    for shard_connections in shard_connections_list:
        for shard_connection in shard_connections:
            shard_id = shard_connection.shard_id
            weight = shard_connection.weight
            if shard_id not in shard_weights:
                shard_weights[shard_id] = []
            shard_weights[shard_id].append(weight)
    avg_weights = {}
    for shard_id, weights in shard_weights.items():
        avg_weights[shard_id] = sum(weights) / len(weights)
    aggregated_shard_connections = [Shard_connection(shard_id, weight) for shard_id, weight in avg_weights.items()]

    return aggregated_shard_connections
