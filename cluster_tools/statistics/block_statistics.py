#! /bin/python

import os
import sys
import json

import luigi
import numpy as np
import nifty.tools as nt

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Node Label Tasks
#

class BlockStatisticsBase(luigi.Task):
    """ BlockStatistics base class
    """

    task_name = 'block_statistics'
    src_file = os.path.abspath(__file__)

    path = luigi.Parameter()
    key = luigi.Parameter()

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'path': self.path,
                       'key': self.key,
                       'tmp_folder': self.tmp_folder,
                       'block_shape': block_shape})

        with vu.file_reader(self.path, 'r') as f:
            shape = f[self.key].shape

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape,
                                             roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)
        n_jobs = min(len(block_list), self.max_jobs)

        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class BlockStatisticsLocal(BlockStatisticsBase, LocalTask):
    """ BlockStatistics on local machine
    """
    pass


class BlockStatisticsSlurm(BlockStatisticsBase, SlurmTask):
    """ BlockStatistics on slurm cluster
    """
    pass


class BlockStatisticsLSF(BlockStatisticsBase, LSFTask):
    """ BlockStatistics on lsf cluster
    """
    pass


#
# Implementation
#

def merge_stats(stat_list):
    sizes = np.array([stat['size'] for stat in stat_list])
    size = np.sum(sizes)

    means = np.array([stat['mean'] for stat in stat_list])
    mean = np.sum(means * sizes) / size

    variances = np.array([stat['variance'] for stat in stat_list])
    var = (sizes * (variances + (means - mean) ** 2)).sum() / size

    max_ = np.max([stat['max'] for stat in stat_list])
    min_ = np.min([stat['min'] for stat in stat_list])

    stats = {
        'mean': mean,
        'max': max_,
        'min': min_,
        'std': np.sqrt(var),
        'variance': var,
        'size': size
    }
    return stats


def _compute_block_stats(block_id, blocking, ds):
    block = blocking.getBlock(block_id)
    bb = vu.block_to_bb(block)

    data = ds[bb]
    var = float(np.var(data))
    stats = {
        'mean': float(np.mean(data)),
        'max': float(np.max(data)),
        'min': float(np.min(data)),
        'std': float(np.sqrt(var)),
        'variance': var,
        'size': float(data.size)
    }
    return stats


def block_statistics(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    path = config['path']
    key = config['key']
    tmp_folder = config['tmp_folder']

    block_shape = config['block_shape']
    block_list = config['block_list']

    with vu.file_reader(path, 'r') as f:
        shape = f[key].shape

    blocking = nt.blocking([0, 0, 0],
                           list(shape),
                           list(block_shape))

    with vu.file_reader(path, 'r') as f_in:
        ds = f_in[key]
        block_stats = [_compute_block_stats(block_id, blocking, ds)
                       for block_id in block_list]

    save_path = os.path.join(tmp_folder, 'block_statistics_job%i.json' % job_id)
    job_stats = merge_stats(block_stats)
    with open(save_path, 'w') as f:
        json.dump(job_stats, f)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    block_statistics(job_id, path)
