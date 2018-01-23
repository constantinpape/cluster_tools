import os
import argparse
from concurrent import futures
import numpy as np
import z5py


def connected_components_ufd(in_path, in_key,
                             out_path, out_key,
                             block_shape, out_chunks,
                             tmp_folder, n_threads):
    assert os.path.exists(in_path)
    assert os.path.exists(out_path)
    n5_in = z5py.File(in_path, use_zarr_format=False)
    ds = n5_in[in_key]
    shape = ds.shape

    n5_out = z5py.File(out_path, use_zarr_format=False)
    ds_out = n5_out.create_dataset(out_key, dtype='uint64',
                                   shape=shape, chunks=out_chunks,
                                   compression='gzip')

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    halo = [1, 1, 1]
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)


if __name__ == '__main__':
    pass
