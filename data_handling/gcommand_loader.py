import os
import os.path
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.utils.data as data

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]

NPY_EXTENSIONS = [
    '.npy'
]


def is_npy_file(filename):
    return any(filename.endswith(extension) for extension in NPY_EXTENSIONS)


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, audio):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if audio:
                    if is_audio_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        spects.append(item)
                else:
                    if is_npy_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        spects.append(item)
    return spects


def make_dataset_test(dir, audio):
    spects = []
    files = []
    dir = os.path.expanduser(dir)

    d = dir  # os.path.join(dir, "")

    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            if audio:
                if is_audio_file(fname):
                    files.append(fname)
                    path = os.path.join(root, fname)
                    item = (path, 0)
                    spects.append(item)
            else:
                if is_npy_file(fname):
                    files.append(fname)
                    path = os.path.join(root, fname)
                    item = (path, 0)
                    spects.append(item)
    return spects, files


def npy_loader(path):
    spect = np.load(path)
    return spect

class GCommandLoader(data.Dataset):
    """A google command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (Path): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: type of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spectrogram (list): List of (spectrogram path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root: Path, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101, audio=True):

        classes, class_to_idx = find_classes(root)
        spectrogram = make_dataset(root, class_to_idx, audio)

        if len(spectrogram) == 0:
            raise (RuntimeError(
                "Found 0 sound files in subfolders of: " + str(root) + "Supported audio file extensions are: " + ",".join(
                    AUDIO_EXTENSIONS)))

        self.root = root
        self.spectrogram = spectrogram
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len
        self.audio = audio

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spectrogram[index]
        if self.audio:
            spect = self.__spect_loader(path)
        else:
            spect = npy_loader(path)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return spect, target, path

    def __len__(self):
        return len(self.spectrogram)

    def __spect_loader(self, path):
        y, sr = librosa.load(path, sr=None)
        # n_fft = 4096
        n_fft = int(sr * self.window_size)
        win_length = n_fft
        hop_length = int(sr * self.window_stride)

        # STFT

        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window_type)
        spect, phase = librosa.magphase(D)

        # S = log(S+1)
        spect = np.log1p(spect)

        # make all spects with the same dims
        # TODO: change that in the future
        if spect.shape[1] < self.max_len:
            pad = np.zeros((spect.shape[0], self.max_len - spect.shape[1]))
            spect = np.hstack((spect, pad))
        elif spect.shape[1] > self.max_len:
            spect = spect[:, :self.max_len]
        spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
        spect = torch.FloatTensor(spect)

        # z-score normalization
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            if std != 0:
                spect.add_(-mean)
                spect.div_(std)

        return spect