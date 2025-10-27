import numpy as np
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
import torch
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform


def compute_models_distance_pca(models_list, n_components=10, batch_size=100):
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


def hierarchical_clustering_with_centroids(dist_matrix, num_shards):
    if isinstance(dist_matrix, torch.Tensor):
        dist_matrix = dist_matrix.numpy()
    condensed_dist_matrix = pdist(dist_matrix)
    Z = linkage(condensed_dist_matrix, method='ward')
    shard_labels = fcluster(Z, t=num_shards, criterion='maxclust')
    shard_labels = np.array(shard_labels) - 1
    shard_labels = torch.tensor(shard_labels)

    return shard_labels


def model_Hierarchical(proj, client_models):
    dist_matrix = compute_models_distance_pca(client_models)
    shard_labels = hierarchical_clustering_with_centroids(dist_matrix, proj.num_shards)
    shard_clients_ids = defaultdict(list)
    for client_id, shard_id in enumerate(shard_labels):
        shard_clients_ids[shard_id.item()].append(client_id)
    for shard_id, current_clients in shard_clients_ids.items():
        print(f"Shard {shard_id + 1} contains clients: {current_clients}")
        print(f"Client count for Shard {shard_id}: {len(current_clients)}")
    print()

    return shard_clients_ids
