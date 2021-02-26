#! /usr/bin/python

import os
import sys
import json

import luigi
import numpy as np

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
    size_threshold = luigi.IntParameter(default=None)
    target_number = luigi.IntParameter(default=None)
    # task that is required before running this task
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        assert (self.size_threshold is None) != (self.target_number is None)
        # get shape and make block config
        shape = vu.get_shape(self.input_path, self.input_key)

        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        config = {'tmp_folder': self.tmp_folder, 'n_jobs': n_jobs}
        if self.size_threshold is not None:
            config['size_threshold'] = self.size_threshold
        else:
            config['target_number'] = self.target_number

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

    unique_values = [np.load(os.path.join(tmp_folder, 'find_uniques_job_%i.npy' % job_id))
                     for job_id in range(n_jobs)]
    count_values = [np.load(os.path.join(tmp_folder, 'counts_job_%i.npy' % job_id))
                    for job_id in range(n_jobs)]
    uniques = np.unique(np.concatenate(unique_values))
    counts = np.zeros(int(uniques[-1]) + 1, dtype='uint64')

    for uniques_job, counts_job in zip(unique_values, count_values):
        counts[uniques_job] += counts_job.astype('uint64')
    counts = counts[uniques]
    assert len(counts) == len(uniques)

    if 'size_threshold' in config:
        size_threshold = config['size_threshold']
        fu.log("applying size filter %i" % size_threshold)
        discard_ids = uniques[counts < size_threshold]
    else:
        target_number = config['target_number']
        fu.log("filtering until we only have %i segments" % target_number)
        size_sorted = np.argsort(counts)[::-1]
        discard_ids = uniques[size_sorted[target_number:]]
    fu.log("discarding %i / %i ids" % (len(discard_ids), len(uniques)))

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
