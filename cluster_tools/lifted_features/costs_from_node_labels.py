#! /bin/python

import os
import sys
import json
import numpy as np

import luigi
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Node Label Tasks
#

class CostsFromNodeLabelsBase(luigi.Task):
    """ CostsFromNodeLabels base class
    """

    # TODO we need to get the attractive and repulsive edge strength
    # edge strength of lifted edges connecting two nodes with the same node label
    intra_label_weight = ''
    # edge strength of lifted edges connecting two nodes with the same node label
    inter_label_weight = ''

    task_name = 'costs_from_node_labels'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    nh_path = luigi.Parameter()
    nh_key = luigi.Parameter()
    node_label_path = luigi.Parameter()
    node_label_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    prefix = luigi.Parameter()
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
        config.update({'nh_path': self.nh_path,
                       'nh_key': self.nh_key,
                       'node_label_path': self.node_label_path,
                       'node_label_key': self.node_label_key,
                       'output_path': self.output_path,
                       'output_key': self.output_key})
        #
        with vu.file_reader(self.nh_path, 'r') as f:
            n_lifted_edges = f[self.nh_key].shape
        edge_chunk_size = min(64**3, n_lifted_edges)
        with vu.file_reader(self.output_path) as f:
            f.require_dataset(self.output_key, shape=(n_lifted_edges,),
                              chunks=(edge_chunk_size,), compression='gzip',
                              dtype='float32')

        # TODO can we split this into chunks ???
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


class CostsFromNodeLabelsLocal(CostsFromNodeLabelsBase, LocalTask):
    """ CostsFromNodeLabels on local machine
    """
    pass


class CostsFromNodeLabelsSlurm(CostsFromNodeLabelsBase, SlurmTask):
    """ CostsFromNodeLabels on slurm cluster
    """
    pass


class CostsFromNodeLabelsLSF(CostsFromNodeLabelsBase, LSFTask):
    """ CostsFromNodeLabels on lsf cluster
    """
    pass


#
# Implementation
#

def costs_from_node_labels(job_id, config_path):

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

    ndist.computeLiftedNeighborhoodFromNodeLabels(os.path.join(graph_path, graph_key),
                                                  os.path.join(node_label_path, node_label_key),
                                                  os.path.join(output_path, output_key),
                                                  graph_depth, n_threads)

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    costs_from_node_labels(job_id, path)
