import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_loss(train_loss, val_loss, val_acc):
    n = len(train_loss)
    epochs = np.linspace(1, n, n)

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(epochs, train_loss, 'g', label='Training loss')
    ax1.plot(epochs, val_loss, 'r', label='Validation loss')

    ax1.set_title('Training and Validation loss')
    ax1.legend()

    ax1.set(xlabel='epochs', ylabel='loss')

    ax2.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set(xlabel='epochs', ylabel='Accuracy')


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
