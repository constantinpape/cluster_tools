#! /bin/python

import os
import sys
import json

import luigi
import nifty.distributed as ndist

import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Node Label Tasks
#

class SparseLiftedNeighborhoodBase(luigi.Task):
    """ SparseLiftedNeighborhood base class
    """

    task_name = 'sparse_lifted_neighborhood'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    node_label_path = luigi.Parameter()
    node_label_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    prefix = luigi.Parameter()
    nh_graph_depth = luigi.IntParameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        config = LocalTask.default_task_config()
        # different modes for adding lifted edges:
        # "all": add lifted edges between all nodes that have a label
        # "same": add lifted edges only between nodes with the same label
        # "different": add lifted edges only between nodes with different labels
        config.update({'mode': 'all'})
        return config

    def run_impl(self):
        # get the global config and init configs
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'graph_path': self.graph_path,
                       'graph_key': self.graph_key,
                       'node_label_path': self.node_label_path,
                       'node_label_key': self.node_label_key,
                       'output_path': self.output_path,
                       'output_key': self.output_key,
                       'nh_graph_depth': self.nh_graph_depth})

        # prime and run the jobs
        self.prepare_jobs(1, None, config, self.prefix)
        self.submit_jobs(1, self.prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs(self.prefix)
        self.check_jobs(1, self.prefix)

    # part of the luigi API
    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_%s.log' % self.prefix))


class SparseLiftedNeighborhoodLocal(SparseLiftedNeighborhoodBase, LocalTask):
    """ SparseLiftedNeighborhood on local machine
    """
    pass


class SparseLiftedNeighborhoodSlurm(SparseLiftedNeighborhoodBase, SlurmTask):
    """ SparseLiftedNeighborhood on slurm cluster
    """
    pass


class SparseLiftedNeighborhoodLSF(SparseLiftedNeighborhoodBase, LSFTask):
    """ SparseLiftedNeighborhood on lsf cluster
    """
    pass


#
# Implementation
#

def sparse_lifted_neighborhood(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    graph_path = config['graph_path']
    graph_key = config['graph_key']
    node_label_path = config['node_label_path']
    node_label_key = config['node_label_key']
    output_path = config['output_path']
    output_key = config['output_key']

    n_threads = config.get('threads_per_job', 1)
    graph_depth = config['nh_graph_depth']

    mode = config.get('mode', 'all')
    fu.log("lifted nh mode set to %s, depth set to %i" % (mode, graph_depth))

    fu.log("start lifted neighborhood extraction for depth %i" % graph_depth)
    ndist.computeLiftedNeighborhoodFromNodeLabels(os.path.join(graph_path, graph_key),
                                                  os.path.join(node_label_path, node_label_key),
                                                  os.path.join(output_path, output_key),
                                                  graph_depth, n_threads, mode)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    sparse_lifted_neighborhood(job_id, path)
