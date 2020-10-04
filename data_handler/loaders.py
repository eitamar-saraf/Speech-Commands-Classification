from torch.utils.data import DataLoader
from data_handler.gcommand_loader import GCommandLoader
from const import Consts


def get_data_loaders(path=None):
    # load the data
    train_dataset = GCommandLoader(path + 'train')
    train_loader = DataLoader(
        train_dataset, batch_size=Consts.batch_size, shuffle=True, num_workers=20,
         pin_memory=True, sampler=None)

    valid_dataset = GCommandLoader(path + 'valid')
    valid_loader = DataLoader(
        valid_dataset, batch_size=Consts.batch_size, shuffle=True, num_workers=2,
        pin_memory=True)
    test_dataset = GCommandLoader(path + 'test', is_test=True)
    test_loader = DataLoader(
        test_dataset, batch_size=Consts.batch_size, shuffle=True, num_workers=2,
        pin_memory=True)
    return train_loader, valid_loader, test_loader

