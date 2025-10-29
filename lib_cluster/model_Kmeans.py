import numpy as np
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
import torch
from sklearn.decomposition import PCA, IncrementalPCA


# compute model distance
def compute_models_distance_pca(models_list, n_components=10, batch_size=50):
    num_models = len(models_list)
    num_batches = (num_models + batch_size - 1) // batch_size
    reduced_matrices = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_models)
        models_batch = models_list[start_idx:end_idx]
        weight_matrices = []
        for model in models_batch:
            weight_matrix = np.concatenate([v.cpu().numpy().flatten() for v in model.values()])
            weight_matrices.append(weight_matrix)
        weight_matrix_batch = np.stack(weight_matrices)
        weight_matrix_batch = torch.tensor(weight_matrix_batch).float()
        pca = PCA(n_components=n_components)
        weight_matrix_reduced_batch = pca.fit_transform(weight_matrix_batch.numpy())
        weight_matrix_reduced_batch = torch.tensor(weight_matrix_reduced_batch).float()
        reduced_matrices.append(weight_matrix_reduced_batch)
        del weight_matrix_batch, pca, weight_matrix_reduced_batch
        torch.cuda.empty_cache()
    weight_matrix_reduced = torch.cat(reduced_matrices, dim=0)

    return weight_matrix_reduced


def cluster_clients_mini_batch(distances, num_shards, batch_size=100):
    for _ in range(10):
        try:
            distances = distances.float()
            kmeans = MiniBatchKMeans(n_clusters=num_shards, random_state=None, n_init=2,
                                     batch_size=batch_size).fit(distances.numpy())
            labels = kmeans.labels_
            if len(set(labels)) == num_shards:
                return labels
        except Exception as e:
            print(f"Clustering failed: {e}")
    raise ValueError("Clustering failed after multiple attempts")


def model_Kmeans(proj, client_models):
    distances_reduced = compute_models_distance_pca(client_models)
    print(1)
    shard_labels = cluster_clients_mini_batch(distances_reduced, proj.num_shards)
    print(2)
    shard_clients_ids = defaultdict(list)
    for client_id, shard_id in enumerate(shard_labels):
        shard_clients_ids[shard_id].append(client_id)
    for shard_id, current_clients in shard_clients_ids.items():
        print(f"Shard {shard_id + 1} contains clients: {current_clients}")
        print(f"Client count for Shard {shard_id}: {len(current_clients)}")
    print()

    return shard_clients_ids
