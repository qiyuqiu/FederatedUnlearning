import random
from lib_getdata.MNIST import get_mnist_FL
from lib_getdata.FMNIST import get_fmnist_FL
from lib_getdata.cifar10 import get_cifar10_FL
from lib_getdata.femnist import get_femnist_FL
from lib_server.FL_server import server
from lib_results_plot.plot_results import results
from attribute_FL import proj

proj = proj()
# proj.local_epochs = 5
proj.is_save_client_model = True

results = results()
results.epochs = proj.epochs
results.title = 'MNIST-CNN-dirichlet-alpha=0.5'

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

print("------------------------------------------------")
print("Training the model using FATS...")
print("------------------------------------------------")
server_FATS = server(proj, clients)
server_FATS.FATS()

# randomly select a client to revoke
client_unlearn_id = random.randint(0, len(server_FATS.clients) - 1)
client_unlearn = server_FATS.clients[client_unlearn_id]

server_FATS.FATS_unlearn(client_unlearn)
results.FATS = server_FATS.acc_test_epoch

results.save_to_file('FATS_results_summary.txt')

# save results
with open('results_intermediate.txt', 'a') as f:
    f.write('FATS Results:\n')
    f.write(str(results.FATS) + '\n\n')
