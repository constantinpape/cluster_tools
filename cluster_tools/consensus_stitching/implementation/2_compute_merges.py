#! /usr/bin/python

import os
import argparse
from concurrent import futures

import vigra
import numpy as np
import nifty
import z5py


def step2(path, out_key, cache_folder, n_jobs,
          merge_threshold, n_threads,
          prefixes):

    # load in parallel - why not
    paths = [os.path.join(cache_folder, '%s_uvs_%i.json' % (prefix, job_id))
             for prefix in prefixes for job_id in range(n_jobs)]
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(np.load, path) for path in paths]
        results = [t.result() for t in tasks]
        uv_ids = np.concatenate([res for res in results if res.size], axis=0)

    paths = [os.path.join(cache_folder, '%s_votes_%i.json' % (prefix, job_id))
             for prefix in prefixes for job_id in range(n_jobs)]
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(np.load, path) for path in paths]
        results = [t.result() for t in tasks]
        votes = np.concatenate([res for res in results if res.size], axis=0)

    # merge the votes of all blocks and compute the vote ratio
    final_uv_ids, final_votes = nifty.tools.mergeMergeVotes(uv_ids, votes)
    vote_ratio = final_votes[:, 0].astype('float64') / final_votes[:, 1]

    # merge all node pairs whose ratio is above the merge threshold
    merges = vote_ratio > merge_threshold
    merge_node_pairs = final_uv_ids[merges]

    n_labels = int(final_uv_ids.max()) + 1
    ufd = nifty.ufd.ufd(n_labels)

    ufd.merge(merge_node_pairs)
    node_labels = ufd.elementLabeling()
    max_id = vigra.analysis.relabelConsecutive(node_labels, keep_zeros=True,
                                               start_label=1, out=node_labels)[1]
    # write the number of labels to file
    ds = z5py.File(path)[out_key]
    ds.atts["maxId"] = max_id

    out_path = os.path.join(cache_folder, 'node_assignments.npy')
    np.save(out_path, node_labels)
    print("Success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('cache_folder', type=str)
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('merge_threshold', type=float)
    parser.add_argument('n_threads', type=int)
    parser.add_argument('prefixes', type=str, nargs='+')
    args = parser.parse_args()

    step2(args.path, args.out_key,
          args.cache_folder, args.n_jobs,
          args.merge_threshold, args.n_threads,
          list(args.prefixes))
