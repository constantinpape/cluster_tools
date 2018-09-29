#! /bin/python

import os
import sys
import json

import numpy as np
import luigi
import nifty
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
import cluster_tools.utils.segmentation_utils as su
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask

#
# Multicut Tasks
#


class InsertBase(luigi.Task):
    """ Insert base class
    """

    task_name = 'insert'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input volumes and graph
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    decomposition_path = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'graph_path': self.graph_path, 'graph_key': self.graph_key,
                       'output_path': self.output_path, 'output_key': self.output_key,
                       'tmp_folder': self.tmp_folder, 'decomposition_path': self.decomposition_path,
                       'n_jobs': self.max_jobs})

        # prime and run the job
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()

        self._write_log('saving results to %s' % self.output_path)
        self._write_log('and key %s' % self.output_key)

        self.check_jobs(1)


class InsertLocal(InsertBase, LocalTask):
    """ Insert on local machine
    """
    pass


class InsertSlurm(InsertBase, SlurmTask):
    """ Insert on slurm cluster
    """
    pass


class InsertLSF(InsertBase, LSFTask):
    """ Insert on lsf cluster
    """
    pass


#
# Implementation
#


# TODO instead of connected components, we could
# also merge with a ufd here, should compare
def insert(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    # path to the reduced problem
    graph_path = config['graph_path']
    graph_key = config['graph_key']
    decomposition_path = config['decomposition_path']
    # path where the node labeling shall be written
    output_path = config['output_path']
    output_key = config['output_key']
    n_threads = config['threads_per_job']

    tmp_folder = config['tmp_folder']
    n_jobs = config['n_jobs']

    # load the graph
    # TODO parallelize ?!
    graph = ndist.Graph(os.path.join(graph_path, graph_key))
    with vu.file_reader(graph_path, 'r') as f:
        ignore_label = f[graph_key].attrs['ignoreLabel']

    # load the cut edges from initial decomposition
    with vu.file_reader(decomposition_path, 'r') as f:
        ds = f['cut_edges']
        ds.n_threads = n_threads
        cut_edges_decomp = ds[:]

    # load all the sub results
    cut_edges = np.concatenate([np.load(os.path.join(tmp_folder, 'subproblem_results', 'job%i.npy' % job_id))
                               for job_id in range(n_jobs)])
    cut_edges = np.unique(cut_edges).astype('uint64')
    cut_edges = np.concatenate([cut_edges_decomp, cut_edges])

    edge_labels = np.zeros(graph.numberOfEdges, dtype='bool')
    edge_labels[cut_edges] = 1

    node_labeling = ndist.connectedComponents(graph, edge_labels, ignore_label)

    n_nodes = len(node_labeling)
    node_shape = (n_nodes,)
    chunks = (min(n_nodes, 524288),)
    with vu.file_reader(output_path) as f:
        ds = f.require_dataset(output_key, dtype='uint64',
                               shape=node_shape,
                               chunks=chunks,
                               compression='gzip')
        ds.n_threads = n_threads
        ds[:] = node_labeling
    fu.log('saving results to %s' % output_path)
    fu.log('and key %s' % output_key)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    insert(job_id, path)
