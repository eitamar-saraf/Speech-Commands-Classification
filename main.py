import os

import rootpath
import torch
from torch import optim
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from const import Consts
from data_handler.loaders import get_data_loaders
from model import LeNet, weight_init
from train import train, evaluation, test_model


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


def main(device):
    path = rootpath.detect()
    path = os.path.join(path, 'dataset/')
    train_loader, valid_loader, test_loader = get_data_loaders(path)

    model = LeNet(35)
    optimizer = optim.Adam(model.parameters(), lr=Consts.lr, weight_decay=1e-6)
    loss_criterion = nn.NLLLoss()
    model.apply(weight_init)
    model.to(device)
    train_loss = []
    val_loss = []
    val_acc = []
    for epoch in range(Consts.epochs):
        t_loss = train(model, train_loader, optimizer, loss_criterion, device)
        v_loss, v_acc = evaluation(model, valid_loader, loss_criterion, device)
        torch.save(model.state_dict(), f'models/epoch-{epoch + 1}.pth')
        train_loss.append(t_loss)
        val_loss.append(v_loss)
        val_acc.append(v_acc)
        print(f'train loss in epoch {epoch + 1} is: {t_loss}')
        print(f'validation loss in epoch {epoch + 1} is: {v_loss}')
        print(f'validation accuracy in epoch {epoch + 1} is: {v_acc}')

    plot_graphs(train_loss, val_loss, val_acc)
    test_model(model, test_loader, loss_criterion, val_loss, device, 'models/')


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    main(device)
