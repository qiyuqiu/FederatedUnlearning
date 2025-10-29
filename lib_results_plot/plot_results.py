import matplotlib.pyplot as plt
import os


class results:
    def __init__(self):
        self.FL = None
        self.FL_HC = None
        self.SFL = None
        self.SFL_connect = None
        self.FedCIO = None
        self.FATS = None
        self.epochs = None
        self.title = None

    def plot(self):
        plt.figure(figsize=(10, 6))
        if self.SFL is not None and len(self.SFL) > 0:
            epochs = list(range(1, len(self.SFL) + 1))
            plt.plot(epochs, self.SFL, label='SFL', color='blue', linestyle=':')
        if self.FL is not None and len(self.FL) > 0:
            epochs = list(range(1, len(self.FL) + 1))
            plt.plot(epochs, self.FL, label='FL', color='red', linestyle='--')
        if self.FL_HC is not None and len(self.FL_HC) > 0:
            epochs = list(range(1, len(self.FL_HC) + 1))
            plt.plot(epochs, self.FL_HC, label='FL+HC', color='green', linestyle='-.')
        if self.SFL_connect is not None and len(self.SFL_connect) > 0:
            epochs = list(range(1, len(self.SFL_connect) + 1))
            plt.plot(epochs, self.SFL_connect, label='SFL-connect', color='purple', linestyle=':', linewidth=2)
        if self.FedCIO is not None and len(self.FedCIO) > 0:
            epochs = list(range(1, len(self.FedCIO) + 1))
            plt.plot(epochs, self.FedCIO, label='FedCIO', color='yellow', linestyle='-')
        if self.FATS is not None and len(self.FATS) > 0:
            epochs = list(range(1, len(self.FATS) + 1))
            plt.plot(epochs, self.FATS, label='FedCIO', color='orange', linestyle='--', linewidth=2)
        plt.title(self.title)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        save_dir = 'results_of_experiment'
        save_path = os.path.join(save_dir, self.title)
        if not save_path.lower().endswith('.png'):
            save_path += '.png'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)
        plt.show()

    def save_to_file(self, file_path):
        with open(file_path, 'w') as file:
            file.write(f"Title: {self.title}\n")
            file.write(f"Epochs: {self.epochs}\n")
            file.write(f"FL: {self.FL}\n")
            file.write(f"FL_HC: {self.FL_HC}\n")
            file.write(f"SFL: {self.SFL}\n")
            file.write(f"SFL_connect: {self.SFL_connect}\n")
            file.write(f"FedCIO: {self.FedCIO}\n")
            file.write(f"FATS: {self.FATS}\n")
