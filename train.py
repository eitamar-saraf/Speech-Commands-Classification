import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from const import Consts
from data_handling.loaders import get_data_loaders, get_test_loader
from model import LeNet, weight_init
from utils import plot_graphs


def train_lenet(device, dataset_path):
    train_loader, valid_loader, test_loader = get_data_loaders(dataset_path)

    model = LeNet(35)
    optimizer = optim.Adam(model.parameters(), lr=Consts.lr, weight_decay=Consts.weight_decay)
    loss_criterion = torch.nn.NLLLoss()
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


def evaluation(model, loader, loss_criterion, device):
    m_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch_x = batch[0]
            batch_y = batch[1]

            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device, dtype=torch.long)

            pred = model(batch_x)
            loss = loss_criterion(pred, batch_y)
            pred = pred.data.max(1, keepdims=True)[1]
            correct += pred.eq(batch_y.data.view_as(pred)).cpu().sum()
            m_loss += loss.item()

        m_loss = m_loss / len(loader)
        print(correct)
        correct = correct.item() / len(loader.dataset)

        return m_loss, correct


def train(model, train_loader, optimizer, loss_criterion, device):
    n_train = len(train_loader)
    m_loss = 0

    model.train()

    with tqdm(total=n_train, unit='commands') as pbar:
        for batch_index, batch in enumerate(train_loader):
            batch_x = batch[0]
            batch_y = batch[1]
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device, dtype=torch.long)
            pred = model(batch_x)
            loss = loss_criterion(pred, batch_y)
            m_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)

        m_loss = m_loss / len(train_loader)
    return m_loss


def test_model(model, test_loader, loss_criterion, val_loss, device, path=None):
    if path:
        best_model_index = np.argmin(val_loss)
        model.load_state_dict(torch.load(f'{path}epoch-{best_model_index}.pth'))

    test_loss, test_acc = evaluation(model, test_loader, loss_criterion, device)
    print(f'test accuracy is: {test_acc}')


def test(device, test_dataset_path, model_path):
    test_loader = get_test_loader(test_dataset_path)
    loss_criterion = torch.nn.NLLLoss()
    model = LeNet(35)
    if device.type == 'cpu':
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model.load_state_dict(torch.load(model_path))
    test_model(model, test_loader, loss_criterion, [], device, None)
