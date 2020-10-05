import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_loss(train_loss, val_loss, val_acc):
    n = len(train_loss)
    epochs = np.linspace(1, n, n)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(epochs, train_loss, 'g', label='Training loss')
    ax1.plot(epochs, val_loss, 'r', label='Validation loss')

    ax1.title('Training and Validation loss')
    ax1.legend()

    ax1.ylabel('loss')
    ax1.xlabel('epochs')

    ax1.show()

    ax2.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    ax2.title('Validation Accuracy')
    ax2.ylabel('Accuracy')
    ax2.xlabel('epochs')
    ax2.show()


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
