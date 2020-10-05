import numpy as np
import torch
from tqdm import tqdm


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


def test_model(model, test_loader, loss_criterion, val_loss, device, path):
    best_model_index = np.argmin(val_loss)
    model.load_state_dict(torch.load(f'{path}epoch-{best_model_index}.pth'))

    test_loss, test_acc = evaluation(model, test_loader, loss_criterion, device)
    print(f'test accuracy is: {test_acc}')
