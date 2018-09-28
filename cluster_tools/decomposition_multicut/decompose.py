#! /bin/python

import os
import sys
import json
from concurrent import futures

import numpy as np
import luigi
import vigra
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Multicut Tasks
#


class DecomposeBase(luigi.Task):
    """ Decompose base class
    """

    task_name = 'decompose'
    src_file = os.path.abspath(__file__)

    # input volumes and graph
    costs_path = luigi.Parameter()
    costs_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    output_path = luigi.Parameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'costs_path': self.costs_path, 'costs_key': self.costs_key,
                       'graph_path': self.graph_path, 'graph_key': self.graph_key,
                       'output_path': self.output_path})

        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class DecomposeLocal(DecomposeBase, LocalTask):
    """ Decompose on local machine
    """
    pass


class DecomposeSlurm(DecomposeBase, SlurmTask):
    """ Decompose on slurm cluster
    """
    pass


class DecomposeLSF(DecomposeBase, LSFTask):
    """ Decompose on lsf cluster
    """
    pass


#
# Implementation
#


def decompose(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    # input configs
    costs_path = config['costs_path']
    costs_key = config['costs_key']
    graph_path = config['graph_path']
    graph_key = config['graph_key']
    output_path = config['output_path']
    n_threads = config['threads_per_job']

    with vu.file_reader(costs_path, 'r') as f:
        ds = f[costs_key]
        ds.n_threads = n_threads
        costs = ds[:]

    with vu.file_reader(graph_path, 'r') as f:
        ignore_label = f[graph_key].attrs['ignoreLabel']

    # load the graph
    # TODO parallelize ?!
    graph = ndist.Graph(os.path.join(graph_path, graph_key))

    # mark repulsive edges as cut
    edge_labels = costs < 0

    # find the connected components
    labels = ndist.connectedComponents(graph, edge_labels, ignore_label)
    labels, max_id, _ = vigra.analysis.relabelConsecutive(labels, keep_zeros=True, start_label=1)

    # find the edges between connected components
    # which will be cut
    uv_ids = graph.uvIds()
    cut_edges = labels[uv_ids[:, 0]] != labels[uv_ids[:, 1]]
    cut_edges = np.where(cut_edges)[0].astype('uint64')

    n_nodes = labels.shape[0]
    node_shape = (n_nodes,)
    node_chunks = (min(n_nodes, 524288),)

    n_edges = cut_edges.shape[0]
    edge_shape = (n_edges,)
    edge_chunks = (min(n_edges, 524288),)

    with vu.file_reader(output_path) as f:
        ds = f.require_dataset('graph_labels', dtype='uint64',
                               shape=node_shape,
                               chunks=node_chunks,
                               compression='gzip')
        ds.n_threads = n_threads
        ds[:] = labels
        ds.attrs['max_id'] = max_id

        ds = f.require_dataset('cut_edges', dtype='uint64',
                               shape=edge_shape,
                               chunks=edge_chunks,
                               compression='gzip')
        ds.n_threads = n_threads
        ds[:] = cut_edges

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    decompose(job_id, path)
