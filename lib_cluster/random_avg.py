import random


def random_shard_assignment(num_users, num_shards):
    shards = {i: [] for i in range(num_shards)}
    clients = list(range(num_users))
    random.shuffle(clients)
    for shard_id in range(num_shards):
        for i in range(len(clients) // num_shards):
            shards[shard_id].append(clients.pop(0))
    for i, client in enumerate(clients):
        shards[i % num_shards].append(client)

    return shards


def random_shard_assignment_avg(num_users, num_shards):
    if num_users % num_shards != 0:
        return random_shard_assignment(num_users, num_shards)
    shards = {i: [] for i in range(num_shards)}
    clients = list(range(num_users))
    random.shuffle(clients)
    shard_size = num_users // num_shards
    for i in range(num_shards):
        start_index = i * shard_size
        end_index = start_index + shard_size
        shards[i] = clients[start_index:end_index]

    return shards
