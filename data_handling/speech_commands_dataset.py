from pathlib import Path
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset


class SpeechCommandsDataset(Dataset):
    def __init__(self, data: List[Tuple[Path, int]]):
        self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_path, label =  self.data[index]
        sample = np.load(sample_path)
        return sample, label

