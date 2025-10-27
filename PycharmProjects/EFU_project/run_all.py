import subprocess
import os
import shutil

# Let the user specify the Python interpreter path
python_interpreter = '/path/to/your/python'  # <-- Replace this with your interpreter path


def run_script(script_name):
    result = subprocess.run([python_interpreter, script_name], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"{script_name} ran successfully.")
    else:
        print(f"Error running {script_name}:\n{result.stderr}")


# Define available datasets
datasets = {
    'MNIST-Dirichlet-alpha=0.5': 'get_mnist_FL(proj, alpha=0.5)',
    'MNIST-pathological-num_label_client=2': 'get_mnist_FL(proj, num_label_client=2)',
    'FMNIST-Dirichlet-alpha=0.5': 'get_fmnist_FL(proj, alpha=0.5)',
    'FMNIST-pathological-num_label_client=2': 'get_fmnist_FL(proj, num_label_client=2)',
    'CIFAR10-Dirichlet-alpha=0.5': 'get_cifar10_FL(proj, alpha=0.5)',
    'CIFAR10-pathological-num_label_client=2': 'get_cifar10_FL(proj, num_label_client=2)',
    'FEMNIST': 'get_femnist_FL(proj)'
}

# User-selected dataset
selected_dataset_key = 'FMNIST-Dirichlet-alpha=0.5'  # You can change this to any other key
selected_dataset = datasets[selected_dataset_key]

# Define output folder
results_folder = 'results_output-' + selected_dataset_key

# Create the results folder if it doesn't exist
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Save the selected dataset to a config file
config_file = 'dataset_config.txt'
with open(config_file, 'w') as f:
    f.write(f"clients = {selected_dataset}\n")

# Scripts to run
scripts = ['run_FLB.py', 'run_FL_HC.py', 'run_EFU.py', 'run_FATS.py']
# scripts = ['run_FATS.py']  # Uncomment to run only one script

# Clear the intermediate results file
intermediate_results_file = os.path.join(results_folder, 'results_intermediate.txt')
with open(intermediate_results_file, 'w') as f:
    f.write("")

# Execute each script
for script in scripts:
    run_script(script)

# Combine all results into a final summary file
final_results_file = os.path.join(results_folder, f'final_results_summary_{selected_dataset_key}.txt')
with open(intermediate_results_file, 'r') as f:
    intermediate_results = f.read()

with open(final_results_file, 'w') as f:
    f.write("Final Results Summary:\n")
    f.write(intermediate_results)

print(f"All scripts executed and results collected in folder '{results_folder}'.")

# Optionally move other result files to the results folder
other_results_files = [
    'FL_results_summary.txt',
    'FL_HC_results_summary.txt',
    'SFL_results_summary.txt',
    'FATS_results_summary.txt'
]
for result_file in other_results_files:
    if os.path.exists(result_file):
        shutil.move(result_file, os.path.join(results_folder, result_file))
    else:
        print(f"{result_file} not found.")
