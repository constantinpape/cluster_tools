#! /usr/bin/python

import json
import time
import argparse
import os
import numpy as np
import subprocess
# from concurrent import futures

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

        f = z5py.File(self.path)
        ds = f[self.out_key]
        shape = ds.shape
        blocking = nifty.tools.blocking([0, 0, 0], shape, block_shape)
        n_blocks = blocking.numberOfBlocks

        n_jobs = min(n_blocks, self.max_jobs)

        # submit the job
        out_path = self.output().path
        command = '%s %s %i %s' % (script_path, self.tmp_folder, n_jobs, out_path)
        log_file = os.path.join(self.tmp_folder, 'logs', 'log_nde_assignment')
        err_file = os.path.join(self.tmp_folder, 'error_logs', 'err_nde_assignment.err')
        bsub_command = 'bsub -J nde_assignment -We %i -o %s -e %s \'%s\'' % (self.time_estimate,
                                                                             log_file, err_file, command)
        if self.run_local:
            subprocess.call([command], shell=True)
        else:
            subprocess.call([bsub_command], shell=True)

        # wait till all jobs are finished
        if not self.run_local:
            util.wait_for_jobs('papec')

        # check for correct execution
        success = os.path.exists(out_path)
        if not success:
            raise RuntimeError("Compute node assignment failed")

    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder, 'node_assignment.npy'))


def compute_node_assignment(tmp_folder, n_jobs, out_path):

    t0 = time.time()
    assignments = []
    # TODO parallelize ?!
    for job_id in range(n_jobs):
        res_path = os.path.join(tmp_folder, 'node_assignments_job%i.npy' % job_id)
        assignment = np.load(res_path)
        if assignment.size > 0:
            assignments.append(assignment)
    assignments = np.concatenate(assignments, axis=0)
    n_labels = int(assignments.max()) + 1

    # stitch the segmentation (node level)
    ufd = nifty.ufd.ufd(n_labels)
    ufd.merge(assignments)
    node_labels = ufd.elementLabeling()
    vigra.analysis.relabelConsecutive(node_labels, keep_zeros=True,
                                      start_label=1, out=node_labels)

    np.save(out_path, node_labels)
    with open(os.path.join(tmp_folder, 'node_assignment_time.json'), 'w') as f:
        json.dump({'time': time.time() - t0}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tmp_folder', type=str)
    parser.add_argument('n_jobs', type=int)
    parser.add_argument('out_path', type=str)
    args = parser.parse_args()
    compute_node_assignment(args.tmp_folder, args.n_jobs, args.out_path)
