import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def crop_n_block_n_scale(data: torch.Tensor, crop_size: int, block_size: int):
    if crop_size % block_size != 0:
        raise ValueError('block_size should be a factor of crop_size')
    if data.shape[1] < crop_size:
        pad = torch.zeros((data.shape[0], crop_size), dtype=data.dtype, device=data.device)
        dmin, dmax = data.min(dim=1, keepdims=True)[0], data.max(dim=1, keepdims=True)[0]
        data = (data - dmin) / (dmax-dmin) - 0.5  # between -0.5 and 0.5
        # scale before padding so pad values remain zero
        pad[:, :data.shape[1]] = data
        data = pad
    else:
        data = data[:, :crop_size]
        # Scale after cropping so the output range isn't affected by the crop
        dmin, dmax = data.min(dim=1, keepdims=True)[0], data.max(dim=1, keepdims=True)[0]
        data = (data - dmin) / (dmax-dmin) - 0.5  # between -0.5 and 0.5
    return data.reshape(data.shape[0], -1, block_size)


class VocalizationDataset(Dataset):
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
        return data
    
    def collate_fn(self, batch):
        batch = [torch.from_numpy(b) for b in batch]
        batch = pad_sequence(batch, batch_first=True)
        # Subtracting one block from the sequence length to avoid having
        # my attention module pad the input
        # I can't figure out why masking operation on padded inputs
        # creates NaN gradients
        batch = crop_n_block_n_scale(
            batch,
            self.crop_size - self.block_size,
            self.block_size
        )
        return batch
    