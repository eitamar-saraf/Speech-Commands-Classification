import os

import rootpath
import numpy as np


from data_handler.loaders import get_datasets


def transform_dataset(loader):
    for k, (x, label, path) in enumerate(loader):
        x = x.numpy()
        new_path = path.replace('wav', 'npy')
        np.save(new_path, x)


def transform():
    path = rootpath.detect()
    path = os.path.join(path, 'dataset/')
    train_loader, valid_loader, test_loader = get_datasets(path)
    transform_dataset(train_loader)
    transform_dataset(valid_loader)
    transform_dataset(test_loader)


if __name__ == '__main__':
    transform()
