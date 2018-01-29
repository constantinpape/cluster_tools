#! /usr/bin/python

import os
import argparse
from math import ceil
import numpy as np

import z5py
import nifty


def blocks_to_jobs(shape, block_shape, n_jobs, tmp_folder):
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    n_blocks = blocking.numberOfBlocks
    assert n_jobs <= n_blocks, "%i, %i" % (n_jobs, n_blocks)
    chunk_size = int(ceil(float(n_blocks) / n_jobs))
    block_list = list(range(n_blocks))
    for idx, i in enumerate(range(0, len(block_list), chunk_size)):
        np.save(os.path.join(tmp_folder, '1_input_%i.npy' % idx), block_list[i:i + chunk_size])
    assert idx == n_jobs - 1, "Not enough inputs created: %i / %i" % (idx, n_jobs - 1)


def prepare(aff_path_xy, key_xy,
            aff_path_z, key_z,
            out_path, key_out,
            out_chunks, out_blocks,
            tmp_folder, n_jobs):
    assert os.path.exists(aff_path_xy)
    assert os.path.exists(aff_path_z)
    assert all(bs % cs == 0 for bs, cs in zip(out_blocks, out_chunks)), \
        "Block shape is not a multiple of chunk shape: %s %s" % (str(out_blocks), str(out_chunks))
    ds_xy = z5py.File(aff_path_xy)[key_xy]
    ds_z = z5py.File(aff_path_z)[key_z]

    shape = ds_xy.shape
    assert ds_z.shape == shape

    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    f_out = z5py.File(out_path, use_zarr_format=False)
    if key_out in f_out:
        ds_out = f_out[key_out]
        assert ds_out.chunks == out_chunks, "%s, %s" % (str(ds_out.chunks), str(out_chunks))
        assert ds_out.shape == shape
    else:
        ds_out = f_out.create_dataset(key_out, shape=shape, chunks=out_chunks, dtype='uint64',
                                      compression='gzip')

    blocks_to_jobs(shape, out_blocks, n_jobs, tmp_folder)
    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("aff_path_xy", type=str)
    parser.add_argument("key_xy", type=str)
    parser.add_argument("aff_path_z", type=str)
    parser.add_argument("key_z", type=str)

    parser.add_argument("out_path", type=str)
    parser.add_argument("out_key", type=str)
    parser.add_argument("--block_shape", nargs=3, type=int)
    parser.add_argument("--chunks", nargs=3, type=int)
    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--n_jobs", type=int)

    args = parser.parse_args()
    prepare(args.aff_path_xy, args.key_xy,
            args.aff_path_z, args.key_z,
            args.out_path, args.out_key,
            tuple(args.chunks), tuple(args.block_shape),
            args.tmp_folder, args.n_jobs)
