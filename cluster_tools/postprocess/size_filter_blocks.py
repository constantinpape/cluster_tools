#! /usr/bin/python

import os
import sys
import json
import pickle

import luigi
import numpy as np
import vigra
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Find Labeling Tasks
#

class SizeFilterBlocksBase(luigi.Task):
    """ SizeFilterBlocks base class
    """

    task_name = 'size_filter_blocks'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    size_threshold = luigi.IntParameter()
    # task that is required before running this task
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)

        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        config = {'input_path': self.input_path, 'input_key': self.input_key,
                  'tmp_folder': self.tmp_folder, 'n_jobs': n_jobs,
                  'size_threshold': self.size_threshold}

        # we only have a single job to find the labeling
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        save_path = os.path.join(self.tmp_folder, 'discard_ids.npy')
        self._write_log("saving results to %s" % save_path)
        self.check_jobs(1)


class SizeFilterBlocksLocal(SizeFilterBlocksBase, LocalTask):
    """
    SizeFilterBlocks on local machine
    """
    pass


class SizeFilterBlocksSlurm(SizeFilterBlocksBase, SlurmTask):
    """
    SizeFilterBlocks on slurm cluster
    """
    pass


class SizeFilterBlocksLSF(SizeFilterBlocksBase, LSFTask):
    """
    SizeFilterBlocks on lsf cluster
    """
    pass



def size_filter_blocks(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    n_jobs = config['n_jobs']
    tmp_folder = config['tmp_folder']
    input_path = config['input_path']
    input_key = config['input_key']
    size_threshold = config['size_threshold']

    # TODO this could be parallelized
    unique_values = np.concatenate([np.load(os.path.join(tmp_folder, 'find_uniques_job_%i.npy' % job_id))
                                    for job_id in range(n_jobs)])
    count_values = np.concatenate([np.load(os.path.join(tmp_folder, 'counts_job_%i.npy' % job_id))
                                   for job_id in range(n_jobs)])
    uniques = nt.unique(unique_values)
    counts = np.zeros(int(uniques[-1]) + 1, dtype='uint64')

    for uniques_job, counts_job in zip(unique_values, count_values):
        counts[uniques_job] += counts_job.astype('uint64')
    counts = counts[counts != 0]
    assert len(counts) == len(uniques)

    discard_ids = uniques[counts < size_threshold]

    save_path = os.path.join(tmp_folder, 'discard_ids.npy')
    fu.log("saving results to %s" % save_path)
    np.save(save_path, discard_ids)
    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    size_filter_blocks(job_id, path)
