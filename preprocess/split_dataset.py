import os
import shutil
from pathlib import Path
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


def move_files(original_fold: Path, data_fold: Path, data_filename_list: Path) -> None:
    with open(data_filename_list) as f:
        for filename in tqdm(f.readlines()):
            splitted_filename = filename.strip().split('/')
            destination_fold = data_fold.joinpath(splitted_filename[0])
            if not destination_fold.exists():
                os.mkdir(destination_fold)
            shutil.move(original_fold.joinpath(filename.strip()), data_fold.joinpath(filename.strip()))


def create_train_fold(original_fold: Path, data_fold: Path, test_fold: Path) -> None:
    # Take only samples
    dir_names = list()
    for folder in test_fold.iterdir():
        if folder.is_dir():
            dir_names.append(folder.name)

    # build train fold
    for folder in original_fold.iterdir():
        if folder.is_dir() and folder.name in dir_names:
            shutil.move(folder, data_fold.joinpath(folder.name))


def split_dataset(speech_commands_folder: Path, out_path: Path) -> None:
    validation_path = speech_commands_folder.joinpath('validation_list.txt')
    test_path = speech_commands_folder.joinpath('testing_list.txt')

    validation_fold = out_path.joinpath('validation')
    test_fold = out_path.joinpath('test')
    train_fold = out_path.joinpath('train')

    for path in [validation_fold, test_fold, train_fold]:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    logger.info('Moving test files')
    move_files(speech_commands_folder, test_fold, test_path)

    logger.info('Moving validation files')
    move_files(speech_commands_folder, validation_fold, validation_path)

    logger.info('Moving train files')
    create_train_fold(speech_commands_folder, train_fold, test_fold)
