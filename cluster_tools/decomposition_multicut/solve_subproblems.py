#! /bin/python

import os
import sys
import json
from concurrent import futures

import numpy as np
import luigi
import nifty
import nifty.tools as nt
import nifty.distributed as ndist

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
import cluster_tools.utils.segmentation_utils as su
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Multicut Tasks
#


class SolveSubproblemsBase(luigi.Task):
    """ SolveSubproblems base class
    """

    task_name = 'solve_subproblems'
    src_file = os.path.abspath(__file__)

    # input volumes and graph
    costs_path = luigi.Parameter()
    costs_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    decomposition_path = luigi.Parameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def clean_up_for_retry(self, block_list):
        super().clean_up_for_retry(block_list)
        # TODO remove any output of failed blocks because it might be corrupted

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'agglomerator': 'kernighan-lin'})
        return config

    def run(self):
        # get the global config and init configs
        self.make_dirs()
        shebang = self.global_config_values()[0]
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'costs_path': self.costs_path, 'costs_key': self.costs_key,
                       'graph_path': self.graph_path, 'graph_key': self.graph_key,
                       'decomposition_path': self.decomposition_path, 'tmp_folder': self.tmp_folder})

        # make folder for the subproblem results
        try:
            os.mkdir(os.path.join(self.tmp_folder, 'subproblem_results'))
        except OSError:
            pass

        if self.n_retries == 0:
            with vu.file_reader(self.decomposition_path, 'r') as f:
                max_id = f['graph_labels'].attrs['max_id']
            block_list = list(range(1, max_id + 1))
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        self.prepare_jobs(n_jobs, block_list, config)
        self.submit_jobs(n_jobs)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs)


class SolveSubproblemsLocal(SolveSubproblemsBase, LocalTask):
    """ SolveSubproblems on local machine
    """
    pass


class SolveSubproblemsSlurm(SolveSubproblemsBase, SlurmTask):
    """ SolveSubproblems on slurm cluster
    """
    pass


class SolveSubproblemsLSF(SolveSubproblemsBase, LSFTask):
    """ SolveSubproblems on lsf cluster
    """
    pass


#
# Implementation
#


def _solve_component(component_id, graph, graph_labels, costs, agglomerator):
    fu.log("start processing block %i" % component_id)

    # get the nodes belonging to the current
    # component
    nodes = np.where(graph_labels == component_id)[0].astype('uint64')

    inner_edges, _, sub_uvs = graph.extractSubgraphFromNodes(nodes)
    assert len(sub_uvs) == len(inner_edges)

    # if we had only a single node (i.e. no edge, return None)
    if len(sub_uvs) == 0:
        fu.log_block_success(component_id)
        return None

    # relabel the sub-uvs for more efficient processing
    sub_uvs, max_id, _ = vigra.analysis.relabelConsecutive(sub_uvs, start_label=0,
                                                           keep_zeros=False)
    n_local_nodes = max_id + 1
    sub_graph = nifty.graph.undirectedGraph(n_local_nodes)
    sub_graph.insertEdges(sub_uvs)

    sub_costs = costs[inner_edges]
    assert len(sub_costs) == sub_graph.numberOfEdges

    sub_result = agglomerator(sub_graph, sub_costs)
    sub_edgeresult = sub_result[sub_uvs[:, 0]] != sub_result[sub_uvs[:, 1]]

    assert len(sub_edgeresult) == len(inner_edges)
    cut_edge_ids = inner_edges[sub_edgeresult]

    fu.log_block_success(component_id)
    return cut_edge_ids


def solve_subproblems(job_id, config_path):

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
    decomposition_path = config['decomposition_path']
    tmp_folder = config['tmp_folder']
    component_list = config['block_list']
    n_threads = config['threads_per_job']
    agglomerator_key = config['agglomerator']

    with vu.file_reader(costs_path, 'r') as f:
        ds = f[costs_key]
        ds.n_threads = n_threads
        costs = ds[:]

    with vu.file_reader(decomposition_path, 'r') as f:
        ds = f['graph_labels']
        ds.n_threads = n_threads
        graph_labels = ds[:]

    # load the graph
    graph = ndist.Graph(os.path.join(graph_path, graph_key),
                        numberOfThreads=n_threads)
    agglomerator = su.key_to_agglomerator(agglomerator_key)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_solve_component,
                           component_id, graph, graph_labels,
                           costs, agglomerator)
                 for component_id in component_list]
        results = [t.result() for t in tasks]

    cut_edge_ids = np.concatenate([res for res in results if res is not None])
    cut_edge_ids = np.unique(cut_edge_ids)

    res_folder = os.path.join(tmp_folder, 'subproblem_results')
    job_res_path = os.path.join(res_folder, 'job%i.npy' % job_id)
    fu.log("saving cut edge results to %s" % job_res_path)
    np.save(job_res_path, cut_edge_ids)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    solve_subproblems(job_id, path)
