from torch.utils.data import DataLoader
from data_handler.gcommand_loader import GCommandLoader
from const import Consts


def get_data_loaders():
    # load the data
    train_dataset = GCommandLoader('dataset/train')
    train_loader = DataLoader(
        train_dataset, batch_size=Consts.batch_size, shuffle=True)

    valid_dataset = GCommandLoader('dataset/valid')
    valid_loader = DataLoader(
        valid_dataset, batch_size=Consts.batch_size, shuffle=True)
    test_dataset = GCommandLoader('dataset/test', is_test=True)
    test_loader = DataLoader(
        test_dataset, batch_size=Consts.batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader
