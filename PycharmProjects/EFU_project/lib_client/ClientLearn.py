import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib_utils.untils import *


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


class LocalUpdate(object):
    def __init__(self, idx, lr, device, BATCH_SIZE, data_train=None, data_test=None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 1
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []

        if len(data_train) == 0:
            self.ldr_train = None
        elif len(data_train) < BATCH_SIZE:
            self.ldr_train = DataLoader(data_train, batch_size=len(data_train), shuffle=True, pin_memory=True)
        else:
            self.ldr_train = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

        if len(data_test) == 0:
            self.ldr_test = None
        elif len(data_test) < BATCH_SIZE:
            self.ldr_test = DataLoader(data_test, batch_size=len(data_test), shuffle=True)
        else:
            self.ldr_test = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)

    def train(self, net):
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        epoch_acc = []
        epoch_loss = []
        for iter_temp in range(self.local_ep):
            batch_acc = []
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                if len(images) < 3:
                    continue
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                fx = net(images)[0]
                loss = self.loss_func(fx, labels)
                acc = calculate_accuracy(fx, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))

        loss_train = sum(epoch_loss) / len(epoch_loss)
        acc_train = sum(epoch_acc) / len(epoch_acc)
        # print(f"Client {self.idx}: Train Loss = {loss_train:.4f}, Train Accuracy = {acc_train:.2f}%")
        return net.state_dict(), loss_train, acc_train

    def evaluate(self, net):
        net.eval()
        epoch_acc = []
        epoch_loss = []
        with torch.no_grad():
            batch_acc = []
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                fx = net(images)[0]
                loss = self.loss_func(fx, labels)
                acc = calculate_accuracy(fx, labels)
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))

        loss_test = sum(epoch_loss) / len(epoch_loss)
        acc_test = sum(epoch_acc) / len(epoch_acc)
        # prGreen(f"Client {self.idx}: Test Loss = {loss_test:.4f}, Test Accuracy = {acc_test:.2f}%")
        return loss_test, acc_test

    # evaluate with Per
    def evaluate_connection(self, shard_connections, num_class):
        all_acc = []
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                probs_accumulate = torch.zeros(images.size(0), num_class, device=self.device)
                total_weight = 0.0
                for sc in shard_connections:
                    if sc.weight == 0:
                        continue
                    _, probs = sc.predict(images, self.device)  # probs 是 softmax 输出
                    probs_accumulate += probs * sc.weight
                    total_weight += sc.weight
                if total_weight > 0:
                    probs_accumulate /= total_weight  # 归一化
                else:
                    continue
                preds = torch.argmax(probs_accumulate, dim=1)
                correct = preds.eq(labels).sum().item()
                acc = 100.0 * correct / labels.size(0)
                all_acc.append(acc)

        acc_test = sum(all_acc) / len(all_acc) if all_acc else 0.0
        return acc_test
