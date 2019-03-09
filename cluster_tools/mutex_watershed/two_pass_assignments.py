#! /usr/bin/python

import os
import sys
import json

import luigi
import vigra
import nifty.ufd as nufd
import nifty.tools as nt
import numpy as np

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Stitching Tasks
#

class TwoPassAssignmentsBase(luigi.Task):
    """ TwoPassAssignments base class
    """

    task_name = 'two_pass_assignments'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    path = luigi.Parameter()
    key = luigi.Parameter()
    assignments_path = luigi.Parameter()
    assignments_key = luigi.Parameter()
    relabel_key = luigi.Parameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape = self.global_config_values()[:2]
        self.init(shebang)

        config = self.get_task_config()
        config.update({'path': self.path, 'key': self.key,
                       'assignments_path': self.assignments_path,
                       'assignments_key': self.assignments_key,
                       'relabel_key': self.relabel_key,
                       'tmp_folder': self.tmp_folder, 'block_shape': block_shape})

        # we only have a single job to find the labeling
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        self.check_jobs(1)


class TwoPassAssignmentsLocal(TwoPassAssignmentsBase, LocalTask):
    """
    TwoPassAssignments on local machine
    """
    pass


class TwoPassAssignmentsSlurm(TwoPassAssignmentsBase, SlurmTask):
    """
    TwoPassAssignments on slurm cluster
    """
    pass


class TwoPassAssignmentsLSF(TwoPassAssignmentsBase, LSFTask):
    """
    TwoPassAssignments on lsf cluster
    """
    pass


def two_pass_assignments(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    path = config['path']
    key = config['key']
    assignments_path = config['assignments_path']
    assignments_key = config['assignments_key']
    relabel_key = config['relabel_key']
    block_shape = config['block_shape']
    tmp_folder = config['tmp_folder']

    with vu.file_reader(path, 'r') as f:
        ds = f[key]
        shape = ds.shape

    blocking = nt.blocking([0, 0, 0], list(shape), block_shape)
    n_blocks = blocking.numberOfBlocks

    # load block assignments
    pattern = os.path.join(tmp_folder, 'mws_two_pass_assignments_block_%i.npy')
    assignments = []
    for block_id in range(n_blocks):
        save_path = pattern % block_id
        # NOTE, we only have assignments for some of the blocks
        # due to checkerboard procesing (and potentially roi)
        if os.path.exists(save_path):
            assignments.append(np.load(save_path))
    assignments = np.concatenate(assignments, axis=0).astype('uint64')
    fu.log("Loaded assignments of shape %s" % str(assignments.shape))

    # load the relabeling and use it to relabel the assignments
    with vu.file_reader(assignments_path, 'r') as f:
        relabeling = f[relabel_key][:]
    # expected format of relabeling:
    # array[n_labels, 2]
    # first column holds the new old ids
    # second column holds the corresponding new (consecutive!) ids
    assert relabeling.ndim == 2
    assert relabeling.shape[1] == 2

    n_labels = len(relabeling)
    old_to_new = dict(zip(relabeling[:, 0], relabeling[:, 1]))
    assignments = nt.takeDict(old_to_new, assignments)
    assert n_labels > assignments.max(), "%i, %i" % (n_labels, assignments.max())

    fu.log("merge %i labels with ufd" % n_labels)
    ufd = nufd.ufd(n_labels)
    ufd.merge(assignments)
    node_labels = ufd.elementLabeling()

    # make sure 0 is mapped to 0
    # TODO should refactor this into util function and use it
    # wherever we need it after ufd labeling
    if node_labels[0] != 0:
        # we have 0 in labels -> need to remap
        if 0 in node_labels:
            node_labels[node_labels == 0] = node_labels.max() + 1
        node_labels[0] = 0

    vigra.analysis.relabelConsecutive(node_labels, out=node_labels, start_label=1, keep_zeros=True)

    with vu.file_reader(assignments_path) as f:
        chunk_size = min(int(1e6), len(node_labels))
        chunks = (chunk_size,)
        ds = f.create_dataset(assignments_key, data=node_labels, compression='gzip',
                              chunks=chunks)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    two_pass_assignments(job_id, path)
