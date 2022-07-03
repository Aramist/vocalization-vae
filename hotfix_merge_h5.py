import glob
import os
from os import path

import h5py
import numpy as np
from tqdm import tqdm


def make_initial_h5(dest_path, init_data):
    with h5py.File(dest_path, 'w') as ctx:
        with h5py.File(init_data, 'r') as init:
            voclen = init['vocalizations'].shape[0]
            idxlen = init['len_idx'].shape[0]
            ctx.create_dataset(
                'vocalizations',
                (voclen,),
                data=init['vocalizations'][:],
                dtype=np.float32,
                maxshape=(None,)
            )
            ctx.create_dataset(
                'len_idx',
                (idxlen,),
                data=init['len_idx'][:],
                dtype=np.int64,
                maxshape=(None,)
            )


def merge_into(existing, new_data):
    with h5py.File(existing, 'r+') as dest:
        with h5py.File(new_data, 'r') as source:
            # Source is to be interpreted here as the place the new data
            # are coming from
            if len(source['len_idx']) < 3:
                return
            voc_dset = dest['vocalizations']
            old_size = voc_dset.shape[0]
            new_size = old_size + source['vocalizations'].shape[0]
            voc_dset.resize((new_size,))
            voc_dset[old_size:] = source['vocalizations'][:]
            updated_idx = source['len_idx'][1:] + dest['len_idx'][-1]

            old_idx_len = dest['len_idx'].shape[0]
            dest['len_idx'].resize((old_idx_len + len(updated_idx),))
            dest['len_idx'][old_idx_len:] = updated_idx


def run_all(h5_dir):
    h5_files = glob.glob(path.join(h5_dir, '*.h5'))
    merged_file = path.join(h5_dir, 'merged.h5')

    make_initial_h5(merged_file, h5_files[0])

    for h5 in tqdm(h5_files[1:]):
        merge_into(merged_file, h5)
        # os.remove(h5)


if __name__ == '__main__':
    run_all('/mnt/ceph/users/atanelus/unlabeled_vocalizations/c3')
