#! /usr/bin/python

import json
import time
import argparse
import os
import numpy as np
import subprocess
from concurrent import futures

import vigra
import nifty
import z5py
import luigi


# TODO multiple threads for ufd ?!
class NodeAssignmentTask(luigi.Task):
    path = luigi.Parameter()
    out_key = luigi.Parameter()
    config_path = luigi.Parameter()
    max_jobs = luigi.IntParameter()
    tmp_folder = luigi.Parameter()
    dependency = luigi.TaskParameter()
    # FIXME default does not work; this still needs to be specified
    time_estimate = luigi.IntParameter(default=10)
    run_local = luigi.BoolParameter(default=False)

    def requires(self):
        return self.dependency

    def run(self):
        from .. import util

        # copy the script to the temp folder and replace the shebang
        file_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(self.tmp_folder, 'node_assignment.py')
        util.copy_and_replace(os.path.join(file_dir, 'node_assignment.py'), script_path)

        # find the number of blocks
        with open(self.config_path) as f:
            config = json.load(f)
            block_shape = config['block_shape']
            n_threads = config['n_threads']

        f = z5py.File(self.path)
        ds = f[self.out_key]
        shape = ds.shape
        blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape)
        n_blocks = blocking.numberOfBlocks

        n_jobs = min(n_blocks, self.max_jobs)

        # prepare the job
        config_path = os.path.join(self.tmp_folder, 'node_assignment_config.json')
        with open(config_path, 'w') as f:
            json.dump({'n_threads': n_threads}, f)
        # submit the job
        command = '%s %s %i %s' % (script_path, self.tmp_folder, n_jobs, config_path)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_node_assignment')
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_node_assignment.err')
        bsub_command = 'bsub -n %i -J nde_assignment -We %i -o %s -e %s \'%s\'' % (n_threads, self.time_estimate,
                                                                                   log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)

        # wait till all jobs are finished
        if not self.run_local:
            util.wait_for_jobs('papec')

        # check for correct execution
        out_path = self.output().path
        success = os.path.exists(out_path)
        if not success:
            raise RuntimeError("Compute node assignment failed")

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'component_assignments.n5', 'assignments'))


def compute_node_assignment(tmp_folder, n_jobs, config_path):

    from production.util import normalize_and_save_assignments
    t0 = time.time()

    with open(config_path) as f:
        n_threads = json.load(f)['n_threads']

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(np.load,
                           os.path.join(tmp_folder, 'node_assignments_job%i.npy' % job_id))
                 for job_id in range(n_jobs)]
        assignments = [t.result() for t in tasks]
    assignments = np.concatenate([ass for ass in assignments if ass.size > 0],
                                 axis=0)
    n_labels = int(assignments.max()) + 1

    # stitch the segmentation (node level)
    ufd = nifty.ufd.ufd(n_labels)
    ufd.merge(assignments)
    node_labels = ufd.elementLabeling()
    vigra.analysis.relabelConsecutive(node_labels, keep_zeros=True,
                                      start_label=1, out=node_labels)
    normalize_and_save_assignments(os.path.join(tmp_folder, 'component_assignments.n5'), 'assignments',
                                   node_labels, n_threads, offset_segment_labels=False)

    with open(os.path.join(tmp_folder, 'node_assignment_time.json'), 'w') as f:
        json.dump({'t': time.time() - t0}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tmp_folder', type=str)
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    compute_node_assignment(args.tmp_folder, args.n_jobs, args.config_path)
