#! /bin/python

import os
import sys
import json
import numpy as np
from concurrent import futures

import luigi
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Node Label Tasks
#

class MergeNodeLabelsBase(luigi.Task):
    """ MergeNodeLabels base class
    """

    task_name = 'merge_node_labels'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    max_overlap = luigi.BoolParameter(default=True)
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        if not self.max_overlap:
            raise NotImplementedError("Only implemented for max overlap")

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path,
                       'input_key': self.input_key,
                       'output_path': self.output_path,
                       'output_key': self.output_key,
                       'max_overlap': self.max_overlap})

        # prime and run the jobs
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class MergeNodeLabelsLocal(MergeNodeLabelsBase, LocalTask):
    """ MergeNodeLabels on local machine
    """
    pass


class MergeNodeLabelsSlurm(MergeNodeLabelsBase, SlurmTask):
    """ MergeNodeLabels on slurm cluster
    """
    pass


class MergeNodeLabelsLSF(MergeNodeLabelsBase, LSFTask):
    """ MergeNodeLabels on lsf cluster
    """
    pass


#
# Implementation
#


def merge_node_labels(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    output_path = config['output_path']
    output_key = config['output_key']
    max_overlap = config['max_overlap']
    n_threads = config.get('threads_per_job', 1)

    # merge and serialize the overlaps
    mergeAndSerializeOverlaps(os.path.join(input_path, input_key),
                              os.path.join(output_path, output_key),
                              max_overlap, numberOfThreads=n_threads)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    merge_node_labels(job_id, path)
