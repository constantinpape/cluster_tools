#! /bin/python

import os
import sys
import json

import luigi
import numpy as np
import nifty.skeletons as nskel

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask
from cluster_tools.utils.task_utils import DummyTask


#
# skeleton eval tasks
#


class SkeletonEvaluationBase(luigi.Task):
    """ SkeletonEvaluation base class
    """

    task_name = 'skeleton_evaluation'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    skeleton_path = luigi.Parameter()
    skeleton_key = luigi.Parameter()
    output_path = luigi.Parameter()
    dependency = luigi.TaskParameter(default=DummyTask())

    @staticmethod
    def default_task_config():
        config = LocalTask.default_task_config()
        # TODO do we need task specific stuff ?
        # config.update({})
        return config

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        # TODO support roi
        shebang, block_shape, _, _ = self.global_config_values()
        self.init(shebang)

        # load the skeleton_evaluation config
        # update the config with input and output paths and keys
        task_config = self.get_task_config()
        task_config.update({'input_path': self.input_path, 'input_key': self.input_key,
                            'skeleton_path': self.skeleton_path, 'skeleton_key': self.skeleton_key,
                            'output_path': self.output_path})

        # prime and run the jobs
        n_jobs = 1
        self.prepare_jobs(n_jobs, None, task_config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class SkeletonEvaluationLocal(SkeletonEvaluationBase, LocalTask):
    """
    skeleton_evaluation on local machine
    """
    pass


class SkeletonEvaluationSlurm(SkeletonEvaluationBase, SlurmTask):
    """
    skeleton_evaluation on slurm cluster
    """
    pass


class SkeletonEvaluationLSF(SkeletonEvaluationBase, LSFTask):
    """
    skeleton_evaluation on lsf cluster
    """
    pass


#
# Implementation
#


def skeleton_evaluation(job_id, config_path):
    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    # read the input cofig
    input_path = config['input_path']
    input_key = config['input_key']

    skeleton_path = config['skeleton_path']
    skeleton_key = config['skeleton_key']

    output_path = config['output_path']
    skeleton_format = config['skeleton_format']
    n_threads = config.get('threads_per_job', 1)

    # TODO adapt nskel.SkeletonMetrics to new n5 skeleton format
    skeleton_ids = os.listdir(skeleton_file)
    skeleton_ids = [int(sk) for sk in skeleton_ids if sk.isdigit()]
    skeleton_ids.sort()
    metrics = nskel.SkeletonMetrics(os.path.join(input_path, input_key),
                                    os.path.join(skeleton_path, skeleton_key),
                                    skeleton_ids, n_threads)

    # TODO expose parameters for different eval options
    correct, split, merge, n_merges = metrics.computeGoogleScore(n_threads)
    res = {'correct': correct, 'split': split, 'merge': merge, 'n_merges': n_merges}
    with open(output_path, 'w') as f:
        json.dump(res, f)

    # log success
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    skeleton_evaluation(job_id, path)
