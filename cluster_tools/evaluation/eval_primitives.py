#! /bin/python

import os
import sys
import json

import luigi
import nifty.distributed as ndist

import cluster_tools.utils.function_utils as fu
import cluster_tools.utils.volume_utils as vu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Evaluation Tasks
#

class EvalPrimitivesBase(luigi.Task):
    """ EvalPrimitives base class
    """

    task_name = 'eval_primitives'
    src_file = os.path.abspath(__file__)
    allow_retries = False

    input_path = luigi.Parameter()
    input_key = luigi.Parameter()
    out_path = luigi.Parameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'input_path': self.input_path,
                       'input_key': self.input_key,
                       'out_path': self.out_path})

        n_jobs = 1

        # prime and run the jobs
        self.prepare_jobs(n_jobs, None, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class EvalPrimitivesLocal(EvalPrimitivesBase, LocalTask):
    """ EvalPrimitives on local machine
    """
    pass


class EvalPrimitivesSlurm(EvalPrimitivesBase, SlurmTask):
    """ EvalPrimitives on slurm cluster
    """
    pass


class EvalPrimitivesLSF(EvalPrimitivesBase, LSFTask):
    """ EvalPrimitives on lsf cluster
    """
    pass


#
# Implementation
#


def eval_primitives(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    input_path = config['input_path']
    input_key = config['input_key']
    out_path = config['out_path']
    n_threads = config.get('threadsPerJob', 1)
    fu.log("Seriailze results to %s" % out_path)

    with vu.file_reader(input_path, 'r') as f:
        attrs = f[input_key].attrs
        n_labels_seg = attrs['n_labels_seg']
        n_labels_gt = attrs['n_labels_gt']
        n_points = attrs['n_points']

    primitives_dict = ndist.computeEvalPrimitives(os.path.join(input_path, input_key),
                                                  n_points, n_labels_seg, n_labels_gt,
                                                  n_threads)
    fu.log("Serializing primitives to %s" % out_path)
    with open(out_path, 'w') as f:
        json.dump(primitives_dict, f)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    eval_primitives(job_id, path)
