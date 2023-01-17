import argparse
from pathlib import Path
import sys
import logging

from preprocess.split_dataset import split_dataset
from preprocess.transform_dataset import transform_dataset_into_spectrogram
from train_test import train, test

logging.basicConfig(format='%(asctime)s, %(levelname)s:, %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    stream=sys.stdout, level=logging.DEBUG)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser that controls the script.')

    parser.add_argument('--action', help='the action that the script should do.',
                        choices=['train', 'create_dataset', 'transform_dataset', 'test'], default='test')
    parser.add_argument('--speech_commands_folder',
                        help='the path to the root folder of te google commands dataset before splitting.',
                        default=r'dataset/raw')
    parser.add_argument('--out_path', help='the path where to save the files splitted to folders.',
                        default='dataset/splitted')
    parser.add_argument('--dataset', help='the path to the root folder of te google commands dataset after splitting.',
                        default='dataset/splitted')
    parser.add_argument('--lr', help='the learning rate.', default=0.0005)
    parser.add_argument('--batch_size', help='the batch size.', default=128)
    parser.add_argument('--epochs', help='the number of epochs.', default=50)
    parser.add_argument('--num_workers', help='the number of workers.', default=8)
    parser.add_argument('--weight_decay', help='the weight decay.', default=0.00001)
    parser.add_argument('--num_classes', help='Number of classes', default=30)
    parser.add_argument('--model_name', help='The name of the model', default='alexnet',
                        choices=['lenet', 'improved_lenet', 'alexnet'])
    parser.add_argument('--model_checkpoint', help='the path to model we want tot test.',
                        default='lightning_logs/AlexNet/checkpoints/speech_commands-epoch=36-validation_weighted_f1=0.967.ckpt')
    args = parser.parse_args()

    if args.action == 'create_dataset':
        logger.info('Creating dataset')
        split_dataset(Path(args.speech_commands_folder), Path(args.out_path))

    elif args.action == 'transform_dataset':
        logger.info('Transforming dataset')
        transform_dataset_into_spectrogram(Path(args.dataset))

    elif args.action == 'train':
        logger.info('Training model')
        train(args)

    elif args.action == 'test':
        logger.info('Testing model')
        test(args)

    else:
        logger.error('Action not recognized')
        raise ValueError('Action not supported')
