#! /bin/python

import os
import sys
import json

import luigi

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.statistics.block_statistics import merge_stats


#
# Node Label Tasks
#

class MergeStatisticsBase(luigi.Task):
    """ MergeStatistics base class
    """

    task_name = 'merge_statistics'
    src_file = os.path.abspath(__file__)
    output_path = luigi.Parameter()
    shape = luigi.ListParameter()
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        block_list = vu.blocks_in_volume(self.shape, block_shape,
                                         roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'output_path': self.output_path,
                       'tmp_folder': self.tmp_folder,
                       'n_jobs': n_jobs})

        # prime and run the jobs
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class MergeStatisticsLocal(MergeStatisticsBase, LocalTask):
    """ MergeStatistics on local machine
    """
    pass


class MergeStatisticsSlurm(MergeStatisticsBase, SlurmTask):
    """ MergeStatistics on slurm cluster
    """
    pass


class MergeStatisticsLSF(MergeStatisticsBase, LSFTask):
    """ MergeStatistics on lsf cluster
    """
    pass


#
# Implementation
#


def merge_statistics(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    n_jobs = config['n_jobs']
    tmp_folder = config['tmp_folder']
    output_path = config['output_path']

    job_stats = []
    for stat_job_id in range(n_jobs):
        job_path = os.path.join(tmp_folder, 'block_statistics_job%i.json' % stat_job_id)
        with open(job_path) as f:
            job_stat = json.load(f)
        job_stats.append(job_stat)

    stats = merge_stats(job_stats)
    with open(output_path, 'w') as f:
        json.dump(stats, f)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    merge_statistics(job_id, path)
