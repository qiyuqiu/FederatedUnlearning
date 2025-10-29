from torch.utils.data import random_split, ConcatDataset
from lib_getdata.DataSplite import *
from lib_client.Client_local import Client
from lib_getdata.DataSplite import DatasetSplit
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Subset
from lib_model.MNIST.ResNet18 import resnet18_MNIST
from lib_model.MNIST.CNN import CNN
from lib_utils.untils import prRed


def get_MNIST(BATCH_SIZE):
    train_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='data',
                                  train=False,
                                  transform=transforms.ToTensor())

    combined_dataset = ConcatDataset([train_dataset, test_dataset])

    return combined_dataset


def get_mnist_FL(proj, num_clients=None, sigma=None, alpha=None, shards_per_user=None,
                 num_label_client=None):
    print("Data:MNIST")

    if num_clients is None:
        num_clients = proj.num_client
    else:
        proj.num_client = num_clients
    proj.num_classes = 10
    # proj.net_glob = resnet18_MNIST(proj.num_classes)
    proj.net_glob = CNN(proj.num_classes)

    total_params = 0
    for param in proj.net_glob.parameters():
        total_params += param.numel()

    proj.plot_title = 'MNIST, CNN.png'
    train_dataset = get_MNIST(proj.BATCH_SIZE)

    if sigma:
        dict_clients = dataset_non_iid_sigma(train_dataset, num_clients, sigma)
    elif alpha:
        dict_clients = dataset_dirichlet_non_iid(train_dataset, num_clients, proj.num_classes)
    elif shards_per_user:
        dict_clients = dataset_pathological_non_iid(train_dataset, num_clients, proj.num_classes)
    elif num_label_client:
        dict_clients = dataset_nonidd_2(train_dataset, proj.num_classes, num_clients, num_label_client)
    else:
        dict_clients = dataset_iid(train_dataset, num_clients)

    clients = []
    for client_id in range(num_clients):
        data_index = dict_clients[client_id]
        data = DatasetSplit(train_dataset, data_index)
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        train_dataset_temp, test_dataset_temp = random_split(data, [train_size, test_size])
        client = Client(client_id, train_dataset_temp, test_dataset_temp, proj)
        clients.append(client)

    for client in clients:
        label_counts = Counter(client.get_train_dataset_labels())
        print(f"Client {client.client_id}: Class distribution - {label_counts}")
        print(f"Client {client.client_id}: Train Dataset Size: {client.get_train_dataset_size()}, "
              f"Test Dataset Size: {client.get_test_dataset_size()}")

    return clients
