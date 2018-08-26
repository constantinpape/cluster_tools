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
    scale = luigi.IntParameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    def clean_up_for_retry(self, block_list):
        # TODO does this work with the mixin pattern?
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
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({'costs_path': self.costs_path, 'costs_key': self.costs_key,
                       'graph_path': self.graph_path, 'graph_key': self.graph_key,
                       'scale': self.scale, 'tmp_folder': self.tmp_folder})

        with vu.file_reader(self.graph_path, 'r') as f:
            shape = f.attrs['shape']

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
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


def _solve_block_problem(block_id, graph, block_prefix, costs, agglomerator):
    fu.log("start processing block %i" % block_id)

    # load the nodes in this sub-block and map them
    # to our current node-labeling
    block_path = block_prefix + str(block_id)
    assert os.path.exists(block_path), block_path
    nodes = ndist.loadNodes(block_path)
    inner_edges, outer_edges, sub_uvs = graph.extractSubgraphFromNodes(nodes)

    # if we had only a single node (i.e. no edge, return the outer edges)
    if len(nodes) == 1:
        return outer_edges

    assert len(sub_uvs) == len(inner_edges)
    assert len(sub_uvs) > 0, str(block_id)

    n_local_nodes = int(sub_uvs.max() + 1)
    sub_graph = nifty.graph.undirectedGraph(n_local_nodes)
    sub_graph.insertEdges(sub_uvs)

    sub_costs = costs[inner_edges]
    assert len(sub_costs) == sub_graph.numberOfEdges

    sub_result = agglomerator(sub_graph, sub_costs)
    sub_edgeresult = sub_result[sub_uvs[:, 0]] != sub_result[sub_uvs[:, 1]]

    assert len(sub_edgeresult) == len(inner_edges)
    cut_edge_ids = inner_edges[sub_edgeresult]
    cut_edge_ids = np.concatenate([cut_edge_ids, outer_edges])

    fu.log_block_success(block_id)
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
    tmp_folder = config['tmp_folder']
    scale = config['scale']
    block_list = config['block_list']
    n_threads = config['threads_per_job']
    agglomerator_key = config['agglomerator']

    if scale == 0:
        block_prefix = os.path.join(graph_path, 'sub_graphs',
                                    's%i' % scale, 'block_')
    else:
        block_prefix = os.path.join(graph_path, 's%i' % scale,
                                    'sub_graphs', 'block_')

    with vu.file_reader(costs_path, 'r') as f:
        ds = f[costs_key]
        ds.n_threads = n_threads
        costs = ds[:]

    # load the graph
    # TODO parallelize ?!
    graph = ndist.Graph(os.path.join(graph_path, graph_key))
    agglomerator = su.key_to_agglomerator(agglomerator_key)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_solve_block_problem,
                           block_id, graph, block_prefix,
                           costs, agglomerator)
                 for block_id in block_list]
        results = [t.result() for t in tasks]

    results = [res for res in results if res is not None]
    if len(results) > 0:
        cut_edge_ids = np.concatenate(results)
        cut_edge_ids = np.unique(cut_edge_ids).astype('uint64')
    else:
        cut_edge_ids = np.zeros(0, dtype='uint64')

    # TODO log save-path
    np.save(os.path.join(tmp_folder, '1_output_s%i_%i.npy' % (scale, job_id)),
            cut_edge_ids)
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    solve_subproblems(job_id, path)
