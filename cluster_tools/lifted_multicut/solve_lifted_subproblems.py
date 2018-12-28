#! /bin/python

import os
import sys
import json
from concurrent import futures

import numpy as np
import vigra
import luigi
import z5py
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
    problem_path = luigi.Parameter()
    scale = luigi.IntParameter()
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
        config.update({'agglomerator': 'kernighan-lin',
                       'time_limit_solver': None})
        return config

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        with vu.file_reader(self.problem_path, 'r') as f:
            shape = tuple(f.attrs['shape'])
            # TODO
            # make sure we have lifted edges
            assert 'lifted_nh' in f

        factor = 2**self.scale
        block_shape = tuple(bs * factor for bs in block_shape)

        # update the config with input and graph paths and keys
        # as well as block shape
        config = self.get_task_config()
        config.update({'problem_path': self.problem_path, 'scale': self.scale,
                       'block_shape': block_shape})

        # make output datasets
        out_key = 's%i/sub_results' % self.scale
        with vu.file_reader(self.problem_path) as f:
            out = f.require_group(out_key)
            # NOTE, gzip may fail for very small inputs, so we use raw compression for now
            # might be a good idea to give blosc a shot ...
            out.require_dataset('cut_edge_ids', shape=shape, chunks=block_shape,
                                compression='raw', dtype='uint64')
            out.require_dataset('node_result', shape=shape, chunks=block_shape,
                                compression='raw', dtype='uint64')

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        else:
            block_list = self.block_list
            self.clean_up_for_retry(block_list)

        n_jobs = min(len(block_list), self.max_jobs)
        # prime and run the jobs
        prefix = 's%i' % self.scale
        self.prepare_jobs(n_jobs, block_list, config, prefix)
        self.submit_jobs(n_jobs, prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(n_jobs, prefix)

    # part of the luigi API
    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_s%i.log' % self.scale))


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


def _solve_block_problem(block_id, graph, uv_ids, block_prefix,
                         costs, agglomerator, ignore_label,
                         blocking, out, time_limit):
    fu.log("Start processing block %i" % block_id)

    # load the nodes in this sub-block and map them
    # to our current node-labeling
    block_path = block_prefix + str(block_id)
    assert os.path.exists(block_path), block_path
    nodes = ndist.loadNodes(block_path)
    # if we have an ignore label, remove zero from the nodes
    # (nodes are sorted, so it will always be at pos 0)
    if ignore_label and nodes[0] == 0:
        nodes = nodes[1:]
        removed_ignore_label = True
        if len(nodes) == 0:
            fu.log_block_success(block_id)
            return
    else:
        removed_ignore_label = False

    # we allow for invalid nodes here,
    # which can occur for un-connected graphs resulting from bad masks ...
    inner_edges, outer_edges = graph.extractSubgraphFromNodes(nodes, allowInvalidNodes=True)

    # if we only have no inner edges, return
    # the outer edges as cut edges
    if len(inner_edges) == 0:
        if len(nodes) > 1:
            assert removed_ignore_label,\
                "Can only have trivial sub-graphs for more than one node if we removed ignore label"
        cut_edge_ids = outer_edges
        sub_result = None
        fu.log("Block %i: has no inner edges" % block_id)
    # otherwise solve the multicut for this block
    else:
        fu.log("Block %i: Solving sub-block with %i nodes and %i edges" % (block_id,
                                                                           len(nodes),
                                                                           len(inner_edges)))
        sub_uvs = uv_ids[inner_edges]
        # relabel the sub-nodes and associated uv-ids for more efficient processing
        nodes_relabeled, max_id, mapping = vigra.analysis.relabelConsecutive(nodes,
                                                                             start_label=0,
                                                                             keep_zeros=False)
        sub_uvs = nt.takeDict(mapping, sub_uvs)
        n_local_nodes = max_id + 1
        sub_graph = nifty.graph.undirectedGraph(n_local_nodes)
        sub_graph.insertEdges(sub_uvs)

        sub_costs = costs[inner_edges]
        assert len(sub_costs) == sub_graph.numberOfEdges

        # solve multicut and relabel the result
        sub_result = agglomerator(sub_graph, sub_costs, time_limit=time_limit)
        assert len(sub_result) == len(nodes), "%i, %i" % (len(sub_result), len(nodes))

        sub_edgeresult = sub_result[sub_uvs[:, 0]] != sub_result[sub_uvs[:, 1]]
        assert len(sub_edgeresult) == len(inner_edges)
        cut_edge_ids = inner_edges[sub_edgeresult]
        cut_edge_ids = np.concatenate([cut_edge_ids, outer_edges])

        _, res_max_id, _ = vigra.analysis.relabelConsecutive(sub_result, start_label=1,
                                                             keep_zeros=False,
                                                             out=sub_result)
        fu.log("Block %i: Subresult has %i unique ids" % (block_id, res_max_id))
        # IMPORTANT !!!
        # we can only add back the ignore label after getting the edge-result !!!
        if removed_ignore_label:
            sub_result = np.concatenate((np.zeros(1, dtype='uint64'),
                                         sub_result))

    # get chunk id of this block
    block = blocking.getBlock(block_id)
    chunk_id = tuple(beg // sh for beg, sh in zip(block.begin, blocking.blockShape))

    # serialize the cut-edge-ids and the (local) node labeling
    ds_edge_res = out['cut_edge_ids']
    fu.log("Block %i: Serializing %i cut edges" % (block_id, len(cut_edge_ids)))
    ds_edge_res.write_chunk(chunk_id, cut_edge_ids, True)

    if sub_result is not None:
        ds_node_res = out['node_result']
        fu.log("Block %i: Serializing %i node results" % (block_id, len(sub_result)))
        ds_node_res.write_chunk(chunk_id, sub_result, True)

    fu.log_block_success(block_id)


def solve_subproblems(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    # input configs
    problem_path = config['problem_path']
    scale = config['scale']
    block_shape = config['block_shape']
    block_list = config['block_list']
    n_threads = config['threads_per_job']
    agglomerator_key = config['agglomerator']
    time_limit = config.get('time_limit_solver', None)

    fu.log("reading problem from %s" % problem_path)
    problem = z5py.N5File(problem_path)
    shape = problem.attrs['shape']

    # load the costs
    costs_key = 's%i/costs' % scale
    fu.log("reading costs from path in problem: %s" % costs_key)
    ds = problem[costs_key]
    ds.n_threads = n_threads
    costs = ds[:]

    # load the graph
    graph_key = 's%i/graph' % scale
    fu.log("reading graph from path in problem: %s" % graph_key)
    graph = ndist.Graph(os.path.join(problem_path, graph_key),
                        numberOfThreads=n_threads)
    uv_ids = graph.uvIds()
    # check if the problem has an ignore-label
    ignore_label = problem[graph_key].attrs['ignoreLabel']
    fu.log("ignore label is %s" % ('true' if ignore_label else 'false'))

    fu.log("using agglomerator %s" % agglomerator_key)
    agglomerator = su.key_to_agglomerator(agglomerator_key)

    # the output group
    out = problem['s%i/sub_results' % scale]

    # TODO this should be a n5 varlen dataset as well and
    # then this is just another dataset in problem path
    block_prefix = os.path.join(problem_path, 's%i' % scale,
                                'sub_graphs', 'block_')
    blocking = nt.blocking([0, 0, 0], shape, list(block_shape))

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_solve_block_problem,
                           block_id, graph, uv_ids, block_prefix,
                           costs, agglomerator, ignore_label,
                           blocking, out, time_limit)
                 for block_id in block_list]
        [t.result() for t in tasks]

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    solve_subproblems(job_id, path)
