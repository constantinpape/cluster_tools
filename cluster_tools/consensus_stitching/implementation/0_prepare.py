#! /usr/bin/python

import os
import argparse
import json

import nifty
import z5py


def make_job_configs(cache_folder, n_jobs, n_blocks, config, prefix):
    assert n_jobs <= n_blocks, "%i, %i" % (n_jobs, n_blocks)
    block_list = list(range(n_blocks))
    for job_id in range(n_jobs):
        block_jobs = block_list[job_id::n_jobs]
        job_config = {'config': config,
                      'blocks': block_jobs}
        with open(os.path.join(cache_folder, '%s_config_%i.json' % (prefix, job_id)), 'w') as f:
            json.dump(job_config, f)


def get_full_offsets():
    return [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [-2, 0, 0], [0, -3, 0], [0, 0, -3],
            [-3, 0, 0], [0, -9, 0], [0, 0, -9],
            [-4, 0, 0], [0, -27, 0], [0, 0, -27]]


def get_nearest_offsets():
    return [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]


def step0(path, ws_key, out_key,
          cache_folder, n_jobs,
          block_shape, block_shift,
          chunks):
    f = z5py.File(path)
    ds = f[ws_key]
    shape = ds.shape

    # require the output dataset
    f.require_dataset(out_key, shape=shape, chunks=chunks,
                      compression='gzip', dtype='uint64')

    # make the tmp folder
    try:
        os.mkdir(cache_folder)
    except OSError:
        pass

    # TODO make parameter
    weight_edges = False

    # make configs for the first blocking
    config1 = {'offsets': get_full_offsets(),
               'block_shape': block_shape,
               'block_shift': None,
               'weight_edges': weight_edges}
    blocking1 = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape))
    n_blocks1 = blocking1.numberOfBlocks
    make_job_configs(cache_folder, n_jobs, n_blocks1,
                     config1, prefix='1_blocking1')

    # make configs for the second blocking
    config2 = {'offsets': get_full_offsets(),
               'block_shape': block_shape,
               'block_shift': block_shift,
               'weight_edges': weight_edges}
    blocking2 = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape),
                                     blockShift=list(block_shift))
    n_blocks2 = blocking2.numberOfBlocks
    make_job_configs(cache_folder, n_jobs, n_blocks2,
                     config2, prefix='1_blocking2')
    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('ws_key', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('cache_folder', type=str)
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('--block_shape', type=int, nargs=3)
    parser.add_argument('--block_shift', type=int, nargs=3)
    parser.add_argument('--chunks', type=int, nargs=3)

    args = parser.parse_args()
    step0(args.path, args.ws_key, args.out_key,
          args.cache_folder, args.n_jobs,
          list(args.block_shape),
          list(args.block_shift),
          tuple(args.chunks))
