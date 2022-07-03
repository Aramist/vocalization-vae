import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class VocalizationDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        self.file = h5py.File(file, 'r')
        
    def __del__(self):
        self.file.close()
        
    def __len__(self):
        return len(self.file['len_idx']) - 1
    
    def __getitem__(self, idx):
        start, end = self.file['len_idx'][idx:idx+2]
        data = self.file['vocalizations'][start:end].astype(np.float32)
        data = (data - data.mean()) / data.std()
        return data
    
    @classmethod
    def collate_fn(cls, batch):
        batch = [torch.from_numpy(b).float() for b in batch]
        return pad_sequence(batch, batch_first=True)