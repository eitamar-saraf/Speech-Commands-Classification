import os

import numpy as np

from data_handling.loaders import get_datasets


def transform_loader(loader):
    for k, (x, label, path) in enumerate(loader):
        x = x.numpy()
        new_path = path.replace('wav', 'npy')
        np.save(new_path, x)
        os.remove(path)


def transform_dataset(dataset_path):
    train_loader, valid_loader, test_loader = get_datasets(dataset_path)
    transform_loader(train_loader)
    transform_loader(valid_loader)
    transform_loader(test_loader)
