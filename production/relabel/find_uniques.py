#! /usr/bin/python

import os
import argparse

import numpy as np
import nifty
import z5py
import luigi


class UniquesTask(luigi.Task):
    """
    """

    path = luigi.Parameter()

    def requires(self):
        pass

    def run(self):
        pass

    def output(self):
        pass


def uniques_in_block(block_id, blocking, ds, tmp_folder):
    block = blocking.getBlock(block_id)
    bb = tuple(slice(b, e) for b, e in zip(block.begin, block.end))
    labels = ds[bb]
    uniques = np.unique(labels)
    np.save(os.path.join(tmp_folder, '1_output_%i.npy' % block_id), uniques)


def relabel_step1(labels_path, labels_key, tmp_folder, block_shape, block_file):
    ds_labels = z5py.File(labels_path, use_zarr_format=False)[labels_key]
    shape = ds_labels.shape

    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    block_list = np.load(block_file)

    [uniques_in_block(block_id, blocking, ds_labels, tmp_folder) for block_id in block_list]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", type=str)
    parser.add_argument("labels_key", type=str)

    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--block_shape", nargs=3, type=int)
    parser.add_argument("--block_file", type=str)

    args = parser.parse_args()
    relabel_step1(args.labels_path, args.labels_key,
                  args.tmp_folder,
                  list(args.block_shape),
                  args.block_file)
