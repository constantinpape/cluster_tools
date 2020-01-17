#! /bin/python

import os
import sys
import json

import luigi
import nifty
from elf.segmentation.clustering import mala_clustering

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask

#
# Agglomerative Clusteing Tasks
#


# TODO support vanilla agglomerative clustering
class AgglomerativeClusteringBase(luigi.Task):
    """ AgglomerativeClustering base class
    """

    task_name = 'agglomerative_clustering'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input volumes and graph
    problem_path = luigi.Parameter()
    features_path = luigi.Parameter()
    features_key = luigi.Parameter()
    assignment_path = luigi.Parameter()
    assignment_key = luigi.Parameter()
    threshold = luigi.FloatParameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    # @staticmethod
    # def default_task_config():
    #     # we use this to get also get the common default config
    #     config = LocalTask.default_task_config()
    #     config.update({})
    #     return config

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'assignment_path': self.assignment_path, 'assignment_key': self.assignment_key,
                       'features_path': self.features_path, 'features_key': self.features_key,
                       'problem_path': self.problem_path, 'threshold': self.threshold})

        # prime and run the job
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class AgglomerativeClusteringLocal(AgglomerativeClusteringBase, LocalTask):
    """ AgglomerativeClustering on local machine
    """
    pass


class AgglomerativeClusteringSlurm(AgglomerativeClusteringBase, SlurmTask):
    """ AgglomerativeClustering on slurm cluster
    """
    pass


class AgglomerativeClusteringLSF(AgglomerativeClusteringBase, LSFTask):
    """ AgglomerativeClustering on lsf cluster
    """
    pass


#
# Implementation
#


def agglomerative_clustering(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    # path to the reduced problem
    problem_path = config['problem_path']
    # path where the node labeling shall be written
    assignment_path = config['assignment_path']
    assignment_key = config['assignment_key']
    features_path = config['features_path']
    features_key = config['features_key']

    threshold = config['threshold']
    n_threads = config['threads_per_job']

    scale = 0
    with vu.file_reader(problem_path) as f:
        group = f['s%i' % scale]
        graph_group = group['graph']
        ignore_label = graph_group.attrs['ignore_label']

        ds = graph_group['edges']
        ds.n_threads = n_threads
        uv_ids = ds[:]
        n_edges = len(uv_ids)

    with vu.file_reader(features_path) as f:
        ds = f[features_key]
        ds.n_threads = n_threads
        edge_features = ds[:, 0].squeeze()
        edge_sizes = ds[:, -1].squeeze()
        assert len(edge_features) == n_edges

    n_nodes = int(uv_ids.max()) + 1
    fu.log("creating graph with %i nodes an %i edges" % (n_nodes, len(uv_ids)))
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)
    fu.log("start agglomeration")
    # TODO also support vanilla agglomerative clustering
    node_labeling = mala_clustering(graph, edge_features, edge_sizes, threshold)
    fu.log("finished agglomeration")

    n_nodes = len(node_labeling)

    # make sure zero is mapped to 0 if we have an ignore label
    if ignore_label and node_labeling[0] != 0:
        new_max_label = int(node_labeling.max() + 1)
        node_labeling[node_labeling == 0] = new_max_label
        node_labeling[0] = 0

    node_shape = (n_nodes,)
    chunks = (min(n_nodes, 524288),)
    with vu.file_reader(assignment_path) as f:
        ds = f.require_dataset(assignment_key, dtype='uint64',
                               shape=node_shape,
                               chunks=chunks,
                               compression='gzip')
        ds.n_threads = n_threads
        ds[:] = node_labeling

    fu.log('saving results to %s:%s' % (assignment_path, assignment_key))
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    agglomerative_clustering(job_id, path)
