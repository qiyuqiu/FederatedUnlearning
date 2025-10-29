import os
import random
from lib_getdata.MNIST import get_mnist_FL
from lib_getdata.FMNIST import get_fmnist_FL
from lib_getdata.cifar10 import get_cifar10_FL
from lib_getdata.femnist import get_femnist_FL
from lib_server.FL_server import server
from lib_results_plot.plot_results import results
from attribute_FL import proj

proj = proj()

results = results()
results.epochs = proj.epochs
results.title = 'CIFAR10'

config_file = 'dataset_config.txt'
with open(config_file, 'r') as f:
    exec(f.read())

# clients = get_mnist_FL(proj, alpha=0.5)
# clients = get_mnist_FL(proj, num_label_client=2)

# clients = get_fmnist_FL(proj, alpha=0.5)
# clients = get_fmnist_FL(proj, num_label_client=2)

# clients = get_cifar10_FL(proj, alpha=0.5)
# clients = get_cifar10_FL(proj, num_label_client=2)

# clients = get_femnist_FL(proj)

# randomly select a client to revoke
# client_unlearn_id = random.randint(0, len(clients) - 1)

print("------------------------------------------------")
print("Training the model using EFU...")
print("------------------------------------------------")
server_SFL = server(proj, clients)
server_SFL.train_SFL()

client_unlearn = server_SFL.shards[1].clients[0]

server_SFL.SFL_unlearn_client(client_unlearn)

results.SFL = server_SFL.acc_test_epoch
results.SFL_connect = server_SFL.acc_shard_test_epoch
results.FedCIO = server_SFL.acc_FedCIO_epoch

results.save_to_file('EFU_results_summary.txt')

# save results
with open('results_intermediate.txt', 'a') as f:
    f.write('EFU Results:\n')
    f.write(str(results.SFL) + '\n')
    f.write(str(results.SFL_connect) + '\n')
    f.write(str(results.FedCIO) + '\n\n')
