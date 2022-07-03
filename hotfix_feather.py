import glob
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def convert_feather(feather_path, idx_path, h5_dir, *, cold_run=True):
    feather_path = Path(feather_path)
    new_name = feather_path.stem + '.h5'
    new_path = Path(h5_dir) / Path(new_name)
    idx = np.load(idx_path)

    df = pd.read_feather(str(feather_path))

    lens = [len(df['audio'].iloc[a]) for a in idx]
    total_len = sum(lens)

    len_idx = np.cumsum(np.array(lens))
    # Insert 0 at the first index
    len_idx = np.insert(len_idx, 0, 0)
    
    with h5py.File(str(new_path), 'w') as ctx:
        dset = ctx.create_dataset(
            "vocalizations",
            (total_len,),
            dtype=np.float32
        )
        for n, (start, end) in enumerate( zip(len_idx[:-1], len_idx[1:]) ):
            if cold_run:
                print(f'Inserting arr of shape {df["audio"].iloc[idx[n]].shape} at dset[{start}:{end}]')
            else:
                dset[start:end] = df['audio'].iloc[idx[n]]
        ctx['len_idx'] = len_idx
    if cold_run:
        os.remove(new_path)
    del df
    if not cold_run:
        # os.remove(feather_path)
        pass
            

def convert_feathers(feather_dir, index_dir):
    feather_files = sorted(glob.glob(os.path.join(feather_dir, '*.feather')))
    npy_names = [
        Path(fn).stem + '.npy'
        for fn in feather_files
    ]
    index_files = [os.path.join(index_dir, fn) for fn in npy_names]
    for feather, index in zip(feather_files, index_files):
        convert_feather(feather, index, feather_dir, cold_run=False)

if __name__ == '__main__':
    convert_feathers(
        '/mnt/home/atanelus/ceph/unlabeled_vocalizations/c3',
        '/mnt/home/atanelus/ceph/unlabeled_vocalizations/c3_idx'
    )