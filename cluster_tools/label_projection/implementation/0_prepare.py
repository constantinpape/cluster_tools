#! /usr/bin/python

import os
import argparse
import numpy as np

import z5py
import nifty


def blocks_to_jobs(shape, block_shape, n_jobs, tmp_folder, roi_begin, roi_end):
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(block_shape))
    if roi_begin is not None:
        assert roi_end is not None
        block_list = blocking.getBlockIdsOverlappingBoundingBox(roi_begin,
                                                                roi_end,
                                                                [0, 0, 0])
        n_blocks = len(block_list)
    else:
        n_blocks = blocking.numberOfBlocks
        block_list = list(range(n_blocks))
    assert n_jobs <= n_blocks, "%i, %i" % (n_jobs, n_blocks)
    for job_id in range(n_jobs):
        np.save(os.path.join(tmp_folder, '1_input_%i.npy' % job_id), block_list[job_id::n_jobs])


def prepare(labels_path, labels_key,
            out_path, out_key,
            tmp_folder, block_shape,
            chunks, n_jobs,
            roi_begin, roi_end):

    assert os.path.exists(labels_path)
    assert all(bs % cs == 0 for bs, cs in zip(block_shape, chunks)), \
        "Block shape is not a multiple of chunk shape: %s %s" % (str(block_shape), str(chunks))
    if roi_begin is not None:
        assert roi_end is not None

    shape = z5py.File(labels_path)[labels_key].shape
    out = z5py.File(out_path, use_zarr_format=False)
    if out_key not in out:
        out.create_dataset(out_key, dtype='uint64', shape=shape,
                           chunks=chunks,
                           compression='gzip')
    else:
        ds = out[out_key]
        assert ds.shape == shape
        assert ds.chunks == chunks

    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)
    blocks_to_jobs(shape, block_shape, n_jobs, tmp_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", type=str)
    parser.add_argument("labels_key", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("out_key", type=str)

    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--block_shape", nargs=3, type=int)
    parser.add_argument("--chunks", nargs=3, type=int)
    parser.add_argument("--n_jobs", type=int)
    parser.add_argument("--roi_begin", type=int, nargs=3, default=None)
    parser.add_argument("--roi_end", type=int, nargs=3, default=None)

    args = parser.parse_args()
    prepare(args.labels_path, args.labels_key,
            args.out_path, args.out_key,
            args.tmp_folder,
            tuple(args.block_shape),
            tuple(args.chunks),
            args.n_jobs,
            list(args.roi_begin),
            list(args.roi_end))
