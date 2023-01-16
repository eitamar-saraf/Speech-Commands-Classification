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


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


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

    def __init__(self, root: Path, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):

        self.root = root
        self.classes, self.class_to_idx = self.__find_classes()
        self.spectrogram = self.__make_dataset()
        if len(self.spectrogram) == 0:
            raise (RuntimeError(
                "Found 0 sound files in subfolders of: " + str(
                    root) + "Supported audio file extensions are: " + ",".join(
                    AUDIO_EXTENSIONS)))

        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spectrogram[index]
        spect = self.__spect_loader(path)
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

    def __find_classes(self):
        classes = [d for d in self.root.iterdir() if self.root.joinpath(d).is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __make_dataset(self):
        spects = []
        directory = self.root.expanduser()
        for target in sorted(directory.iterdir()):
            d = directory.joinpath(target)
            if not d.is_dir():
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_audio_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, self.class_to_idx[target])
                        spects.append(item)

        return spects
