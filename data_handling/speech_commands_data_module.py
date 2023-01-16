from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import logging

from data_handling.speech_commands_dataset import SpeechCommandsDataset

logger = logging.getLogger(__name__)


class SpeechCommandsDataModule(pl.LightningDataModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        self.dataset = Path(args.dataset)
        self.train_dataset_path = Path(args.dataset).joinpath('train')
        self.validation_dataset_path = Path(args.dataset).joinpath('validation')
        self.test_dataset_path = Path(args.dataset).joinpath('test')
        self.data = {'train': [], 'validation': [], 'test': []}

    def prepare_data(self) -> None:
        logger.info('Preparing data')
        self.classes, self.class_to_idx = self.__find_classes()
        for folder in [self.train_dataset_path, self.validation_dataset_path, self.test_dataset_path]:
            print(f'Preparing {folder.name} data')
            # iterate over all folder
            for class_folder in folder.iterdir():
                if class_folder.is_dir():
                    # iterate over all files
                    for file in class_folder.iterdir():
                        if file.is_file() and file.suffix == '.npy':
                            target_class = file.parent.name
                            item = (file, self.class_to_idx[target_class])

                            dataset_it_belongs_to = file.parent.parent.name
                            self.data[dataset_it_belongs_to].append(item)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_dataset = SpeechCommandsDataset(self.data['train'])
            self.validation_dataset = SpeechCommandsDataset(self.data['validation'])
        elif stage == 'test':
            self.test_dataset = SpeechCommandsDataset(self.data['test'])
        else:
            raise ValueError(f'Unknown stage: {stage}')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                          num_workers=self.args.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_dataset, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers)

    def __find_classes(self):
        classes = [d.name for d in self.train_dataset_path.iterdir() if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
