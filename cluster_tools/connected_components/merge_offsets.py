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

class MergeOffsetsBase(luigi.Task):
    """ MergeOffsets base class
    """

    task_name = 'merge_offsets'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    shape = luigi.ListParameter()
    save_path = luigi.Parameter()
    # task that is required before running this task
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        block_list = vu.blocks_in_volume(self.shape, block_shape,
                                         roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        block_list = vu.blocks_in_volume(shape, block_shape,
                                         roi_begin, roi_end)

        n_jobs = min(len(block_list), self.max_jobs)

        config = self.get_task_config()
        config.update({'tmp_folder': self.tmp_folder, 'n_jobs': n_jobs,
                       'save_path': self.save_path})

        # we only have a single job to find the labeling
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        # log the save-path again
        self.check_jobs(1)


class MergeOffsetsLocal(MergeOffsetsBase, LocalTask):
    """
    MergeOffsets on local machine
    """
    pass


class MergeOffsetsSlurm(MergeOffsetsBase, SlurmTask):
    """
    MergeOffsets on slurm cluster
    """
    pass


class MergeOffsetsLSF(MergeOffsetsBase, LSFTask):
    """
    MergeOffsets on lsf cluster
    """
    pass


def merge_offsets(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    tmp_folder = config['tmp_folder']
    n_jobs = config['tmp_folder']
    save_path = config['save_path']

    offsets = {}
    for job_id in range(n_jobs):
        path = os.path.join(tmp_folder,
                            'connected_components_offsets_%i.json' % job_id)
        with open(path, 'r') as f:
            offsets.update(json.load(f))
        os.remove(path)

    offset_list = np.array([v for _, v in sorted(offsets.items())],
                           dtype='uint64')
    empty_blocks = np.where(offset_list == 0)[0].tolist()

    offset_list = np.roll(offset_list, 1)
    offset_list[0] = 0
    offset_list = np.cumsum(offset_list).tolist()

    save_path = os.path.join(tmp_folder, 'block_offsets_%i.json' % job_id)
    with open(save_path, 'w') as f:
        json.dump({'offsets': offset_list,
                   'empty_blocks': empty_blocks}, f)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    merge_offsets(job_id, path)
