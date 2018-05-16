#! /usr/bin/python

import os
import argparse
import json

import nifty
import z5py


# TODO implement different job configs for different regions
def make_adaptive_job_config(cache_folder, n_jobs, n_blocks, block_shape):
    pass


def make_job_configs(cache_folder, n_jobs, n_blocks, block_shape, parameters):
    assert n_jobs <= n_blocks, "%i, %i" % (n_jobs, n_blocks)
    block_list = list(range(n_blocks))
    for job_id in range(n_jobs):
        block_jobs = block_list[job_id::n_jobs]
        block_config = {block_id: parameters for block_id in block_jobs}
        job_config = {'block_shape': block_shape,
                      'block_config': block_config}
        with open(os.path.join(cache_folder, '1_config_%s.json' % job_id), 'w') as f:
            json.dump(job_config, f)


def step0(path, aff_key, out_key,
          cache_folder, n_jobs, block_shape):
    f = z5py.File(path)
    ds = f[aff_key]
    shape = ds.shape

    # require the ootput dataset
    chunks = tuple(bs // 2 for bs in block_shape)
    f.require_dataset(out_key, shape=shape, chunks=chunks,
                      compression='gzip', dtype='uint64')

    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape))
    n_blocks = blocking.numberOfBlocks

    # TODO this should be different depending on block location
    default_parameters = {'boundary_threshold': .2,
                          'distance_threshold': 40.,
                          'sigma': 2.6,
                          'resolution': (40., 4., 4.),
                          'aff_slices': [[0, 3], [12, 13]],
                          'invert_channels': [True, False]}

    make_job_configs(cache_folder, n_jobs, n_blocks, block_shape,
                     default_parameters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('aff_key', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('cache_folder', type=str)
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('block_shape', type=int, nargs=3)

    args = parser.parse_args()
    step0(args.path, args.aff_key, args.out_key,
          args.cache_folder, args.n_lobs, list(args.block_shape))
