#! /usr/bin/python

import os
import time
import json
import argparse
from concurrent import futures
import subprocess

import vigra
import numpy as np
import nifty
import z5py
import luigi


# TODO computation with rois
class MergesTask(luigi.Task):
    """
    Run all thresholding tasks
    """

    # path to the n5 file and keys
    path = luigi.Parameter()
    out_key = luigi.Parameter()
    # maximal number of jobs that will be run in parallel
    max_jobs = luigi.IntParameter()
    # path to the configuration
    config_path = luigi.Parameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)
    # TODO optional parameter to just run a subset of blocks

    def requires(self):
        return self.dependency

    def run(self):
        from .. import util

        # copy the script to the temp folder and replace the shebang
        file_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(self.tmp_folder, 'compute_merges.py')
        util.copy_and_replace(os.path.join(file_dir, 'compute_merges.py'),
                              script_path)

        with open(self.config_path) as f:
            config = json.load(f)
            block_shape = config['block_shape2']
            n_threads = config['n_threads']
            # TODO support computation with roi
            if 'roi' in config:
                have_roi = True

        # find the shape and number of blocks
        f5 = z5py.File(self.path)
        shape = f5[self.out_key].shape
        blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape)
        n_blocks = blocking.numberOfBlocks
        n_jobs = min(n_blocks, self.max_jobs)

        command = '%s %s %s %i %s %s' % (script_path, self.path, self.out_key,
                                         n_jobs, self.config_path, self.tmp_folder)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_merges.log')
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_merges.err')
        bsub_command = 'bsub -n %i -J compute_merges -We %i -o %s -e %s \'%s\'' % (n_threads,
                                                                                   self.time_estimate,
                                                                                   log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)
            util.wait_for_jobs('papec')

        out_path = self.output().path
        success = os.path.exists(out_path)

        if not success:
            raise RuntimeError('MergesTask failed')

    def output(self):
        out_path = os.path.join(self.tmp_folder, 'node_assignments_consensus_stitching.npy')
        return luigi.LocalTarget(out_path)


def compute_merges(path, out_key, n_jobs, config_path, tmp_folder):

    t0 = time.time()
    prefixes = ['a', 'b']
    with open(config_path) as f:
        config = json.load(f)
        merge_threshold = config['merge_threshold']
        n_threads = config['n_threads']

    # load in parallel - why not
    paths = [os.path.join(tmp_folder, 'compute_merge_uvs_%s_%i.npy' % (prefix, job_id))
             for prefix in prefixes for job_id in range(n_jobs)]
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(np.load, path) for path in paths]
        results = [t.result() for t in tasks]
        uv_ids = np.concatenate([res for res in results if res.size], axis=0)

    paths = [os.path.join(tmp_folder, 'compute_merge_votes_%s_%i.npy' % (prefix, job_id))
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
    ds.attrs["maxId"] = max_id

    out_path = os.path.join(tmp_folder, 'node_assignments_consensus_stitching.npy')
    np.save(out_path, node_labels)

    res_path = os.path.join(tmp_folder, 'time_consensus_stitching.json')
    with open(res_path, 'w') as f:
        json.dump({'t': time.time() - t0}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out_key', type=str)
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('config_path', type=str)
    parser.add_argument('tmp_folder', type=str)
    args = parser.parse_args()

    compute_merges(args.path, args.out_key,
                   args.n_jobs, args.config_path,
                   args.tmp_folder)
