#! /bin/python

import os
import sys
import json
import luigi

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask

#
# Clear Labels Tasks
#


# FIXME this behaves a bit weird when run as cluster job
# (job is not properly cancelled)
class ClearLiftedEdgesFromLabelsBase(luigi.Task):
    """ ClearLiftedEdgesFromLabels base class
    """

    task_name = 'clear_lifted_edges_from_labels'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    node_labels_path = luigi.Parameter()
    node_labels_key = luigi.Parameter()
    lifted_edge_path = luigi.Parameter()
    lifted_edge_key = luigi.Parameter()
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
        config.update({'node_labels_path': self.node_labels_path,
                       'node_labels_key': self.node_labels_key,
                       'lifted_edge_path': self.lifted_edge_path,
                       'lifted_edge_key': self.lifted_edge_key})

        # prime and run the job
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class ClearLiftedEdgesFromLabelsLocal(ClearLiftedEdgesFromLabelsBase, LocalTask):
    """ ClearLiftedEdgesFromLabels on local machine
    """
    pass


class ClearLiftedEdgesFromLabelsSlurm(ClearLiftedEdgesFromLabelsBase, SlurmTask):
    """ ClearLiftedEdgesFromLabels on slurm cluster
    """
    pass


class ClearLiftedEdgesFromLabelsLSF(ClearLiftedEdgesFromLabelsBase, LSFTask):
    """ ClearLiftedEdgesFromLabels on lsf cluster
    """
    pass


#
# Implementation
#


def clear_lifted_edges_from_labels(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)
    # get the config
    with open(config_path) as f:
        config = json.load(f)

    node_labels_path = config['node_labels_path']
    node_labels_key = config['node_labels_key']
    lifted_edge_path = config['lifted_edge_path']
    lifted_edge_key = config['lifted_edge_key']
    n_threads = config.get('threads_per_job', 1)

    # load node labels
    with vu.file_reader(node_labels_path, 'r') as f:
        ds = f[node_labels_key]
        ds.n_threads = n_threads
        node_labels = ds[:]
    # load lifted edges
    f = vu.file_reader(lifted_edge_path)
    ds = f[lifted_edge_key]
    ds.n_threads = n_threads
    lifted_edges = ds[:]
    chunks = ds.chunks

    n_lifted = len(lifted_edges)
    # map lifted edges to node labels
    lifted_mapped = node_labels[lifted_edges]
    # mask out lifted edges that mix labels
    lifted_mask = lifted_mapped[:, 0] == lifted_mapped[:, 1]
    lifted_edges = lifted_edges[lifted_mask]

    n_lifted_new = len(lifted_edges)
    fu.log("clear number of lifted edges from %i to %i" % (n_lifted, n_lifted_new))

    # resave lifted edges iff number of edges differs
    if n_lifted_new < n_lifted:
        # remove the old dataset
        del f[lifted_edge_key]
        ds = f.create_dataset(lifted_edge_key, shape=lifted_edges.shape, chunks=chunks,
                              compression='gzip', dtype=lifted_edges.dtype)
        ds.n_threads = n_threads
        ds[:] = lifted_edges

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    clear_lifted_edges_from_labels(job_id, path)
