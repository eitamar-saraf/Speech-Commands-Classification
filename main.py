import argparse
from pathlib import Path
import sys
import logging

from data_handling.create_dataset import make_dataset
from data_handling.transform_dataset import transform_dataset
from train import train_lenet, test
from utils import get_device


logging.basicConfig(format='%(asctime)s, %(levelname)s:, %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    stream=sys.stdout, level=logging.DEBUG)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser that controls the script.')

    parser.add_argument('--action', help='the action that the script should do.',
                        choices=['train', 'test', 'create_dataset', 'transform_dataset'], default='transform_dataset')
    parser.add_argument('--speech_commands_folder',
                        help='the path to the root folder of te google commands dataset before splitting.',
                        default=r'dataset/raw')
    parser.add_argument('--out_path', help='the path where to save the files splitted to folders.',
                        default='dataset/splitted')
    parser.add_argument('--dataset', help='the path to the root folder of te google commands dataset after splitting.',
                        default='dataset/splitted')
    parser.add_argument('--test_dataset', help='the path to the root folder of the test dataset.')
    parser.add_argument('--model', help='the path to the model.')

    args = parser.parse_args()

    if args.action == 'create_dataset':
        logger.info('Creating dataset')
        make_dataset(Path(args.speech_commands_folder), Path(args.out_path))

    elif args.action == 'transform_dataset':
        logger.info('Transforming dataset')
        transform_dataset(Path(args.dataset))

    elif args.action == 'train':
        logger.info('Training model')
        device = get_device()
        train_lenet(device, args.dataset)

    elif args.action == 'test':
        logger.info('Testing model')
        device = get_device()
        test(device, args.test_dataset, args.model)

    else:
        logger.error('Action not recognized')
        raise ValueError('Action not supported')
