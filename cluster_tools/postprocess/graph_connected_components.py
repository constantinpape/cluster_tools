#! /bin/python

import os
import sys
import json
import luigi
import nifty
import vigra

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask

PAINTERA_IGNORE_ID = 18446744073709551615


#
# Connected Components Tasks
#

class GraphConnectedComponentsBase(luigi.Task):
    """ GraphConnectedComponents base class
    """

    task_name = 'graph_connected_components'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    problem_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    assignment_path = luigi.Parameter()
    assignment_key = luigi.Parameter()
    output_path = luigi.Parameter()
    output_key = luigi.Parameter()
    dependency = luigi.TaskParameter()

    @staticmethod
    def default_task_config():
        config = LocalTask.default_task_config()
        config.update({'paintera_hack': False, 'ignore_label': True})
        return config

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
        config.update({'problem_path': self.problem_path,
                       'graph_key': self.graph_key,
                       'assignment_path': self.assignment_path,
                       'assignment_key': self.assignment_key,
                       'output_key': self.output_key,
                       'output_path': self.output_path})

        n_jobs = 1
        # prime and run the jobs
        self.prepare_jobs(n_jobs, None, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class GraphConnectedComponentsLocal(GraphConnectedComponentsBase, LocalTask):
    """ GraphConnectedComponents on local machine
    """
    pass


class GraphConnectedComponentsSlurm(GraphConnectedComponentsBase, SlurmTask):
    """ GraphConnectedComponents on slurm cluster
    """
    pass


class GraphConnectedComponentsLSF(GraphConnectedComponentsBase, LSFTask):
    """ GraphConnectedComponents on lsf cluster
    """
    pass


#
# Implementation
#

def graph_connected_components(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)

    problem_path = config['problem_path']
    graph_key = config['graph_key']
    assignment_path = config['assignment_path']
    assignment_key = config['assignment_key']
    output_path = config['output_path']
    output_key = config['output_key']
    n_threads = config.get('threads_per_job', 1)
    paintera_hack = config.get('paintera_hack', False)
    ignore_label = config.get('ignore_label', True)

    with vu.file_reader(assignment_path, 'r') as f:
        ds_ass = f[assignment_key]
        ds_ass.n_threads = n_threads
        assignments = ds_ass[:]
        chunks = ds_ass.chunks

    with vu.file_reader(problem_path, 'r') as f:
        g = f[graph_key]
        ds = g['edges']
        ds.n_threads = n_threads
        edges = ds[:]

    max_id = int(edges.max())
    exclude_pt_ignore = paintera_hack and max_id == PAINTERA_IGNORE_ID
    if exclude_pt_ignore:
        edge_mask = (edges != PAINTERA_IGNORE_ID).all(axis=1)
        edges = edges[edge_mask]
        max_id = int(edges.max())

    graph = nifty.graph.undirectedGraph(max_id + 1)
    graph.insertEdges(edges)
    assignments = nifty.graph.connectedComponentsFromNodeLabels(graph, assignments, ignoreBackground=ignore_label)
    vigra.analysis.relabelConsecutive(assignments, out=assignments, start_label=1,
                                      keep_zeros=True)

    fu.log("Found %i number of components" % int(assignments.max()))

    with vu.file_reader(output_path) as f:
        ds_out = f.require_dataset(output_key, shape=assignments.shape,
                                   chunks=chunks, compression='gzip',
                                   dtype='uint64')
        ds_out.n_threads = n_threads
        ds_out[:] = assignments

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    graph_connected_components(job_id, path)
