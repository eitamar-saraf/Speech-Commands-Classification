import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_graphs(train_loss, val_loss, val_acc):
    n = len(train_loss)
    epochs = np.linspace(1, n, n)
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.plot(epochs, val_acc, 'r', label='validation accuracy')

    plt.title('Training and Validation loss')
    plt.legend()

    plt.ylabel('loss')
    plt.xlabel('epochs')

    plt.show()


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
