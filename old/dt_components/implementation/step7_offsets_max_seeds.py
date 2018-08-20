#! /usr/bin/python

import os
import argparse
import json
import numpy as np


def step7_offsets(cache_folder, n_jobs):
    offset_dict = {}
    for job_id in range(n_jobs):
        offset_file = os.path.join(cache_folder, '6_results_%i.json' % job_id)
        with open(offset_file) as f:
            offset_dict.update(json.load(f))

    # json turns all keys to str, which messes up sorting
    # hence, we need to transform to int before sorting and then cast
    # back to str when reading from the dict
    block_ids = list(map(int, offset_dict.keys()))
    block_ids.sort()
    offsets = np.array([offset_dict[str(block_id)] for block_id in block_ids], dtype='uint64')

    last_offset = offsets[-1]
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)
    n_labels = int(offsets[-1] + last_offset + 1)

    out_file = os.path.join(cache_folder, 'max_seed_offsets.json')
    with open(out_file, 'w') as f:
        out_dict = {'offsets': offsets.tolist(),
                    'n_labels': n_labels}
        json.dump(out_dict, f)
    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cache_folder', type=str)
    parser.add_argument('n_jobs', type=int)
    args = parser.parse_args()
    step7_offsets(args.cache_folder, args.n_jobs)
