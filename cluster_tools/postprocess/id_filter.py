#! /bin/python

import os
import sys
import json
import numpy as np
import luigi

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Filter Tasks
#

class IdFilterBase(luigi.Task):
    """ IdFilter base class
    """

    task_name = 'id_filter'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    node_label_path = luigi.Parameter()
    node_label_key = luigi.Parameter()
    output_path = luigi.Parameter()
    filter_labels = luigi.ListParameter()
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
        config.update({'node_label_path': self.node_label_path,
                       'node_label_key': self.node_label_key,
                       'filter_labels': self.filter_labels,
                       'output_path': self.output_path})

        n_jobs = 1
        # prime and run the jobs
        self.prepare_jobs(n_jobs, None, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class IdFilterLocal(IdFilterBase, LocalTask):
    """ IdFilter on local machine
    """
    pass


class IdFilterSlurm(IdFilterBase, SlurmTask):
    """ IdFilter on slurm cluster
    """
    pass


class IdFilterLSF(IdFilterBase, LSFTask):
    """ IdFilter on lsf cluster
    """
    pass


#
# Implementation
#

def id_filter(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    node_label_path = config['node_label_path']
    node_label_key = config['node_label_key']
    output_path = config['output_path']
    filter_labels = np.array(config['filter_labels'], dtype='uint64')

    with vu.file_reader(node_label_path, 'r') as f:
        node_labels = f[node_label_key][:]

    # find the node ids that overlap with the filter labels
    filter_mask = np.in1d(node_labels, filter_labels)
    filter_ids = np.where(filter_mask)[0].tolist()

    fu.log("%i ids will be filtered" % len(filter_ids))
    fu.log("saving filter ids to %s" % output_path)
    with open(output_path, 'w') as f:
        json.dump(filter_ids, f)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    id_filter(job_id, path)
