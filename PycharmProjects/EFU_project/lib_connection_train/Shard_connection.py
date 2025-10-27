import torch


class Shard_connection:
    def __init__(self, shard_id, weight):
        self.weight = weight
        self.shard_id = shard_id
        self.model = None

    def predict(self, data, device):
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            outputs, prob = self.model(data)
        return outputs, prob


def predict_connections(shard_connections, images, device):
    total_outputs = None
    total_weight = 0.0
    for sc in shard_connections:
        output, prob = sc.predict(images, device)
        if total_outputs is None:
            total_outputs = sc.weight * prob
        else:
            total_outputs += sc.weight * prob
        total_weight += sc.weight
    total_outputs /= total_weight

    return total_outputs
