#! /usr/bin/python

import os
import argparse
import numpy as np

import z5py
import nifty


def blocks_to_jobs(shape, block_shape, n_jobs, tmp_folder, output_prefix):
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=block_shape)
    n_blocks = blocking.numberOfBlocks
    assert n_jobs <= n_blocks, "%i, %i" % (n_jobs, n_blocks)
    block_list = list(range(n_blocks))
    for job_id in range(n_jobs):
        np.save(os.path.join(tmp_folder, '%s_%i.npy' % (output_prefix, job_id)),
                block_list[job_id::n_jobs])


def prepare(labels_path, labels_key, graph_path, n_jobs, n_scales, tmp_folder, block_shape):
    assert os.path.exists(labels_path), labels_path
    labels = z5py.File(labels_path)[labels_key]
    shape = labels.shape
    f_graph = z5py.File(graph_path, use_zarr_format=False)
    f_graph.attrs['shape'] = shape

    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    # make blocks to jobs for initial graphs
    blocks_to_jobs(shape, block_shape, n_jobs, tmp_folder, '1_input')

    # make blocks to jobs for the additional scale levels (if any)
    if(n_scales == 1):
        print("Success")
        return

    for scale in range(1, n_scales):
        factor = 2**scale
        scale_shape = [bs*factor for bs in block_shape]
        scale_prefix = '2_input_s%i' % scale
        blocks_to_jobs(shape, scale_shape, n_jobs, tmp_folder, scale_prefix)
    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", type=str)
    parser.add_argument("labels_key", type=str)
    parser.add_argument("graph_path", type=str)
    parser.add_argument("--n_jobs", type=int)
    parser.add_argument("--n_scales", type=int)
    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--block_shape", nargs=3, type=int)
    args = parser.parse_args()

    prepare(args.labels_path, args.labels_key,
            args.graph_path, args.n_jobs,
            args.n_scales, args.tmp_folder,
            list(args.block_shape))
