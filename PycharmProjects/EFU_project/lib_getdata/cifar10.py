from torch.utils.data import random_split, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from lib_getdata.DataSplite import *
from lib_client.Client_local import Client
from lib_model.CIFAR.ResNet18 import resnet18_CIFAR10, CIFAR10ResNet18
from lib_getdata.DataSplite import DatasetSplit
from lib_model.CIFAR.CNN import CIFAR_CNN

# Relative path to CIFAR-10 dataset
cifar_dataset_dir_CIFAR10 = 'data/CIFAR-10'


def get_CIFAR10(BATCH_SIZE, cifar_dataset_dir=cifar_dataset_dir_CIFAR10):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = ImageFolder(root=cifar_dataset_dir + '/train', transform=transform_train)
    test_dataset = ImageFolder(root=cifar_dataset_dir + '/val', transform=transform_test)

    combined_dataset = ConcatDataset([train_dataset, test_dataset])
    return combined_dataset


def get_cifar10_FL(proj, num_clients=None, sigma=None, alpha=None, shards_per_user=None, num_label_client=None):
    print("Data: CIFAR-10")

    if num_clients is None:
        num_clients = proj.num_client
    else:
        proj.num_client = num_clients

    proj.num_classes = 10
    proj.epochs = 200

    proj.net_glob = CIFAR_CNN(proj.num_classes)
    # Alternative models:
    # proj.net_glob = resnet18_CIFAR10(proj.num_classes)
    # proj.net_glob = CIFAR10ResNet18(proj.num_classes)

    proj.lr_weight_train = 1
    proj.epochs_weights_train_FL = 1
    proj.epochs_weights_train = 10

    proj.plot_title = 'CIFAR-10, CNN.png'
    proj.lr = 0.0001
    proj.lr_pretrain = proj.lr
    proj.base_cross_loss = 10

    data_all = get_CIFAR10(proj.BATCH_SIZE)

    if sigma:
        dict_clients = dataset_non_iid_sigma(data_all, num_clients, sigma)
    elif alpha:
        dict_clients = dataset_dirichlet_non_iid(data_all, num_clients, proj.num_classes, alpha)
    elif shards_per_user:
        dict_clients = dataset_pathological_non_iid(data_all, num_clients, proj.num_classes)
    elif num_label_client:
        dict_clients = dataset_nonidd_2(data_all, proj.num_classes, num_clients, num_label_client)
    else:
        dict_clients = dataset_iid(data_all, num_clients)

    clients = []
    for client_id in range(num_clients):
        data_index = dict_clients[client_id]
        data = DatasetSplit(data_all, data_index)

        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size

        train_dataset_temp, test_dataset_temp = random_split(data, [train_size, test_size])
        client = Client(client_id, train_dataset_temp, test_dataset_temp, proj)
        clients.append(client)

    for client in clients:
        print(f"Client {client.client_id}: Train Dataset Size: {client.get_train_dataset_size()}, "
              f"Test Dataset Size: {client.get_test_dataset_size()}")

    return clients
