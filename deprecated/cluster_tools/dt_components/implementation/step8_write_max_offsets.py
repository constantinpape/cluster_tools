#! /usr/bin/python

import os
import json
import argparse
import z5py
import nifty
import numpy as np


def write_offsets(ds, blocking, block_id, extended_offset, offsets):
    off = offsets[block_id]
    block = blocking.getBlock(block_id)
    bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
    seg = ds[bb]

    old_max = seg.max()

    # we mask out all the extended seeds
    # (and original mask) and add the offset
    mask = seg > extended_offset
    print("Mask-size:", np.sum(mask))
    seg[mask] += off
    ds[bb] = seg

    new_max = seg.max()
    print("Added offset", off, "to block", block_id, " max value", old_max, "to", new_max)


def step8_write_offsets(path, out_key, cache_folder, job_id):

    offsets1_path = os.path.join(cache_folder, 'block_offsets.json')
    with open(offsets1_path) as f:
        offset_config = json.load(f)
        extended_seed_offset = offset_config['n_labels']
        empty_blocks = offset_config['empty_blocks']

    offsets2_path = os.path.join(cache_folder, 'max_seed_offsets.json')
    with open(offsets2_path) as f:
        offset_config = json.load(f)
        offsets = offset_config['offsets']

    input_file = os.path.join(cache_folder, '1_config_%i.json' % job_id)
    with open(input_file) as f:
        input_config = json.load(f)
        block_shape = input_config['block_shape']
        block_ids = list(input_config['block_config'].keys())
        # json keys are always str, so we need to cast to int
        block_ids = list(map(int, block_ids))

    f = z5py.File(path)
    ds = f[out_key]
    shape = ds.shape
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(block_shape))


    [write_offsets(ds, blocking, block_id, extended_seed_offset, offsets)
     for block_id in block_ids if block_id not in empty_blocks]

    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('cache_folder', type=str)
    parser.add_argument('job_id', type=int)

    args = parser.parse_args()
    step8_write_offsets(args.path, args.out_key, args.cache_folder, args.job_id)
