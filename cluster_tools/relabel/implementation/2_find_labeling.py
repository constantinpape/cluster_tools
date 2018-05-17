#! /usr/bin/python

import os
import argparse
import pickle

import numpy as np
import vigra
import z5py
import nifty


# TODO this could be parallelized
def relabel_step2(labels_path, labels_key, tmp_folder, block_shape, n_threads=1):
    ds_labels = z5py.File(labels_path, use_zarr_format=False)[labels_key]
    shape = ds_labels.shape

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)

    uniques = np.concatenate([np.load(os.path.join(tmp_folder, '1_output_%i.npy' % block_id))
                              for block_id in range(blocking.numberOfBlocks)])
    uniques = np.unique(uniques).astype('uint64', copy=False)
    _, max_id, mapping = vigra.analysis.relabelConsecutive(uniques, keep_zeros=True, start_label=1)

    ds_labels.attrs['maxId'] = max_id

    with open(os.path.join(tmp_folder, '2_output.pkl'), 'wb') as f:
        pickle.dump(mapping, f)

    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", type=str)
    parser.add_argument("labels_key", type=str)

    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--block_shape", nargs=3, type=int)
    parser.add_argument("--n_threads", type=int, default=1)

    args = parser.parse_args()
    relabel_step2(args.labels_path, args.labels_key,
                  args.tmp_folder,
                  list(args.block_shape),
                  args.n_threads)
