#! /usr/bin/python

import os
import argparse
import numpy as np

import z5py
import nifty


def blocks_to_jobs(shape, block_shape, n_jobs, tmp_folder):
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    n_blocks = blocking.numberOfBlocks
    assert n_jobs <= n_blocks, "%i, %i" % (n_jobs, n_blocks)

    block_list = list(range(n_blocks))
    for job_id in range(n_jobs):
        np.save(os.path.join(tmp_folder, '1_input_%i.npy' % job_id),
                block_list[job_id::n_jobs])


def prepare(labels_path, labels_key,
            tmp_folder, block_shape, n_jobs):
    assert os.path.exists(labels_path), labels_path
    ds_labels = z5py.File(labels_path)[labels_key]
    shape = ds_labels.shape

    print(tmp_folder)
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    blocks_to_jobs(shape, block_shape, n_jobs, tmp_folder)
    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", type=str)
    parser.add_argument("labels_key", type=str)

    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--block_shape", nargs=3, type=int)
    parser.add_argument("--n_jobs", type=int)

    args = parser.parse_args()
    prepare(args.labels_path, args.labels_key,
            args.tmp_folder,
            tuple(args.block_shape),
            args.n_jobs)
