from random import randint

import h5py
import numpy as np
from torch.utils.data import Dataset


class LocalizationDataset(Dataset):
    def __init__(self, file, crop_size, block_size):
        super().__init__()
        self.file = h5py.File(file, 'r')
        self.crop_size = crop_size
        self.block_size = block_size
        
    def __del__(self):
        self.file.close()
        
    def __len__(self):
        return len(self.file['len_idx']) - 1
    
    def __getitem__(self, idx):
        start, end = self.file['len_idx'][idx:idx+2]
        data = self.file['vocalizations'][start:end].astype(np.float32)
        data = (data - data.mean()) / data.std()
        return self.sample(data).reshape(-1, data.shape[1] * self.block_size)

    def sample(self, data):
        valid_range = 0, len(data) - self.crop_size - 1
        if valid_range[1] < 0:
            return data
        start_idx = randint(*valid_range)
        return data[start_idx:start_idx+self.crop_size]