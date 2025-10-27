import torch
from torch.utils.data import DataLoader


class Shard_Weights_trainer:
    def __init__(self, proj, shard_models, shard_weights, current_shard_weight_id):
        self.proj = proj
        self.device = proj.device
        self.num_classes = proj.num_classes
        self.net_glob = proj.net_glob
        self.shard_models = shard_models
        self.shard_weights = shard_weights
        self.epochs = proj.epochs_weights_train
        self.lr = proj.lr_weight_train
        self.current_shard_weight_id = current_shard_weight_id
        self.active_shards = {i: True for i in shard_weights.keys()}

    # get cross loss
    def get_crossloss_shardweights(self, images, labels):
        batch_size = images.size(0)
        final_predictions = torch.zeros(batch_size, self.num_classes, device=self.device)
        for shard_id in self.shard_weights.keys():
            if self.active_shards[shard_id]:
                shard_model = self.shard_models[shard_id].to(self.device)
                shard_weight = self.shard_weights[shard_id].to(self.device)
                shard_model.eval()
                with torch.no_grad():
                    outputs, _ = shard_model(images)
                    predictions = torch.softmax(outputs, dim=1)
                final_predictions += predictions * shard_weight
        final_predictions /= torch.sum(
            torch.stack([weight for key, weight in self.shard_weights.items() if self.active_shards[key]]))
        loss = torch.nn.functional.cross_entropy(final_predictions, labels)
        return final_predictions, loss

    # optimal A{}
    def train_shard_weights(self, data_train):
        self.shard_weights = {key: value.clone().detach().requires_grad_(True) if key != 0 else value.clone().detach()
                              for key, value in self.shard_weights.items()}
        optimizer = torch.optim.SGD([value for key, value in self.shard_weights.items() if key != 0], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, min_lr=1e-6)
        # last_valid_weights = {key: value.clone() for key, value in self.shard_weights.items()}
        min_total_loss = float('inf')
        best_weights = None
        for epoch in range(self.epochs):
            total_loss = 0.0
            data_train_loader = DataLoader(data_train, batch_size=len(data_train), shuffle=False)
            for images, labels in data_train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                _, loss = self.get_crossloss_shardweights(images, labels)
                modified_loss = torch.pow(self.proj.base_cross_loss, loss)
                reg_loss = sum(torch.norm(weight) ** 2 for weight in self.shard_weights.values())
                reg_lr = 0.001
                except_loss = sum(torch.norm(weight) ** 2 for key, weight in self.shard_weights.items()
                                  if key != self.current_shard_weight_id and self.active_shards[key])
                except_lr = 0.01
                total_loss_value = modified_loss + reg_lr * reg_loss + except_lr * except_loss
                optimizer.zero_grad()
                total_loss_value.backward()
                optimizer.step()
                total_loss += total_loss_value.item()

            negative_weights = [key for key, weight in self.shard_weights.items() if torch.any(weight < 0)]
            if negative_weights:
                for key in negative_weights:
                    with torch.no_grad():
                        self.shard_weights[key].zero_()
                    self.active_shards[key] = False
                zero_weights_count = sum(torch.sum(weight == 0).item() for weight in self.shard_weights.values())
                total_weights_count = sum(weight.numel() for weight in self.shard_weights.values())
                zero_weights_ratio = zero_weights_count / total_weights_count
                if zero_weights_ratio >= self.proj.num_shard_connection_fraction:
                    return self.shard_weights
            scheduler.step(total_loss)

        return self.shard_weights
