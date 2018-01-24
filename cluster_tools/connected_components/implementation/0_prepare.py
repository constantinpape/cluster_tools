#! /usr/bin/python

import os
import argparse
import nifty
import z5py
import numpy as np


def blocks_to_jobs(shape, block_shape, n_jobs, tmp_folder):
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    n_blocks = blocking.numberOfBlocks
    chunk_size = n_blocks // n_jobs
    block_list = list(range(n_blocks))
    for i in range(0, len(block_list), chunk_size):
        np.save(block_list[i:i + chunk_size], '1_input_%i.npy' % i)


def prepare(in_path, in_key,
            out_path, out_key,
            tmp_folder, out_block_shape,
            out_chunks, n_jobs):

    assert all(bs % cs == 0 for bs, cs in zip(out_block_shape, out_chunks)), "Block shape is not a multiple of chunk shape"
    assert os.path.exists(in_path), "Input at %s does not exist" % in_path
    n5_in = z5py.File(in_path, use_zarr_format=False)
    ds = n5_in[in_key]
    shape = ds.shape

    n5_out = z5py.File(out_path, use_zarr_format=False)
    if out_key not in n5_out:
        ds_out = n5_out.create_dataset(out_key, dtype='uint64',
                                       shape=shape, chunks=out_chunks,
                                       compression='gzip')
    else:
        ds_out = n5_out[out_key]
        assert ds_out.shape == shape
        assert ds_out.chunks == out_chunks

    blocks_to_jobs(shape, out_block_shape, n_jobs, tmp_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", type=str)
    parser.add_argument("in_key", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("out_key", type=str)
    parser.add_argument("--tmp_folder", str)
    parser.add_argument("--block_shape", nargs=3, type=int)
    parser.add_argument("--chunks", nargs=3, type=int)
    parser.add_argument("--n_jobs", type=int)

    args = parser.parse_args()
    prepare(args.in_path, args.in_key,
            args.out_path, args.out_key,
            args.tmp_folder, args.block_shape,
            args.chunks, args.n_jobs)
