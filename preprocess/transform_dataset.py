import os
from pathlib import Path
import logging
import numpy as np

from data_handling.gcommand_loader import GCommandLoader

logger = logging.getLogger(__name__)


def transform_loader(loader):
    for k, (x, label, path) in enumerate(loader):
        x = x.numpy()
        new_path = path.replace('wav', 'npy')
        np.save(new_path, x)
        os.remove(path)


def transform_dataset_into_spectrogram(dataset_path: Path) -> None:
    for folder in dataset_path.iterdir():
        if folder.is_dir():
            logger.info(f'Transforming {folder.name} into spectrogram')

            dataset_loader = GCommandLoader(folder)
            transform_loader(dataset_loader)
