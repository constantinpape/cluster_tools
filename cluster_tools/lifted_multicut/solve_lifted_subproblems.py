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
from elf.segmentation.lifted_multicut import get_lifted_multicut_solver
from elf.segmentation.multicut import get_multicut_solver

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask


#
# Lifted Multicut Tasks
#


class SolveLiftedSubproblemsBase(luigi.Task):
    """ SolveLiftedSubproblems base class
    """

    task_name = 'solve_lifted_subproblems'
    src_file = os.path.abspath(__file__)

    # input volumes and graph
    problem_path = luigi.Parameter()
    lifted_prefix = luigi.Parameter()
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
        # shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        shebang, block_shape, roi_begin, roi_end, block_list_path\
            = self.global_config_values(with_block_list_path=True)
        self.init(shebang)

        with vu.file_reader(self.problem_path, 'r') as f:
            shape = tuple(f['s0/graph'].attrs['shape'])

        factor = 2**self.scale
        block_shape = tuple(bs * factor for bs in block_shape)

        # update the config with input and graph paths and keys
        # as well as block shape
        config = self.get_task_config()
        config.update({'problem_path': self.problem_path, 'scale': self.scale,
                       'block_shape': block_shape, 'lifted_prefix': self.lifted_prefix})

        # make output datasets
        out_key = 's%i/sub_results_lmc' % self.scale
        with vu.file_reader(self.problem_path) as f:
            out = f.require_group(out_key)
            # NOTE, gzip may fail for very small inputs, so we use raw compression for now
            # might be a good idea to give blosc a shot ...
            out.require_dataset('cut_edge_ids', shape=shape, chunks=block_shape,
                                compression='raw', dtype='uint64')
            out.require_dataset('node_result', shape=shape, chunks=block_shape,
                                compression='raw', dtype='uint64')

        if self.n_retries == 0:
            block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end,
                                             block_list_path)
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


class SolveLiftedSubproblemsLocal(SolveLiftedSubproblemsBase, LocalTask):
    """ SolveLiftedSubproblems on local machine
    """
    pass


class SolveLiftedSubproblemsSlurm(SolveLiftedSubproblemsBase, SlurmTask):
    """ SolveLiftedSubproblems on slurm cluster
    """
    pass


class SolveLiftedSubproblemsLSF(SolveLiftedSubproblemsBase, LSFTask):
    """ SolveLiftedSubproblems on lsf cluster
    """
    pass


#
# Implementation
#

def _find_lifted_edges(lifted_uv_ids, node_list):
    lifted_indices = np.arange(len(lifted_uv_ids), dtype='uint64')
    # find overlap of node_list with u-edges
    inner_us = np.in1d(lifted_uv_ids[:, 0], node_list)
    inner_indices = lifted_indices[inner_us]
    inner_uvs = lifted_uv_ids[inner_us]
    # find overlap of node_list with v-edges
    inner_vs = np.in1d(inner_uvs[:, 1], node_list)
    return inner_indices[inner_vs]


def _solve_block_problem(block_id, graph, uv_ids, ds_nodes,
                         costs, lifted_uvs, lifted_costs,
                         lifted_solver, solver,
                         ignore_label, blocking, out, time_limit):
    fu.log("Start processing block %i" % block_id)

    # load the nodes in this sub-block and map them
    # to our current node-labeling
    chunk_id = blocking.blockGridPosition(block_id)
    nodes = ds_nodes.read_chunk(chunk_id)
    if nodes is None:
        fu.log_block_success(block_id)
        return

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
        # find  the lifted uv-ids that correspond to the inner edges
        inner_lifted_edges = _find_lifted_edges(lifted_uvs, nodes)
        fu.log("Block %i: Solving sub-block with %i nodes, %i edges and %i lifted edges" % (block_id,
                                                                                            len(nodes),
                                                                                            len(inner_edges),
                                                                                            len(inner_lifted_edges)))
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

        # we only need to run lifted multicut if we have lifted edges in
        # the subgraph
        if len(inner_lifted_edges) > 0:
            fu.log("Block %i: have lifted edges and use lifted multicut solver" % block_id)
            sub_lifted_uvs = nt.takeDict(mapping, lifted_uvs[inner_lifted_edges])
            sub_lifted_costs = lifted_costs[inner_lifted_edges]

            # solve multicut and relabel the result
            sub_result = lifted_solver(sub_graph, sub_costs, sub_lifted_uvs, sub_lifted_costs,
                                       time_limit=time_limit)

        # otherwise we run normal multicut
        else:
            fu.log("Block %i: don't have lifted edges and use multicut solver")
            # solve multicut and relabel the result
            sub_result = solver(sub_graph, sub_costs, time_limit=time_limit)

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


def solve_lifted_subproblems(job_id, config_path):

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

    lifted_prefix = config['lifted_prefix']
    agglomerator_key = config['agglomerator']
    time_limit = config.get('time_limit_solver', None)
    n_threads = config.get('threads_per_job', 1)

    fu.log("reading problem from %s" % problem_path)
    problem = z5py.N5File(problem_path)

    # load the costs
    # NOTE we use different cost identifiers for multicut and lifted multicut
    # in order to run both in the same n5-container.
    # However, for scale level 0 the costs come from the CostsWorkflow and
    # hence the identifier is identical
    costs_key = 's%i/costs_lmc' % scale if scale > 0 else 's0/costs'
    fu.log("reading costs from path in problem: %s" % costs_key)
    ds = problem[costs_key]
    ds.n_threads = n_threads
    costs = ds[:]

    # load the graph
    # NOTE we use different graph identifiers for multicut and lifted multicut
    # in order to run both in the same n5-container.
    # However, for scale level 0 the graph comes from the GraphWorkflow and
    # hence the identifier is identical
    graph_key = 's%i/graph_lmc' % scale if scale > 0 else 's0/graph'
    shape = problem[graph_key].attrs['shape']

    fu.log("reading graph from path in problem: %s" % graph_key)
    graph = ndist.Graph(problem_path, graph_key, numberOfThreads=n_threads)
    uv_ids = graph.uvIds()
    # check if the problem has an ignore-label
    ignore_label = problem[graph_key].attrs['ignore_label']
    fu.log("ignore label is %s" % ('true' if ignore_label else 'false'))

    fu.log("using agglomerator %s" % agglomerator_key)
    lifted_solver = get_lifted_multicut_solver(agglomerator_key)
    # TODO enable different multicut agglomerator
    solver = get_multicut_solver(agglomerator_key)

    # load the lifted edges and costs
    nh_key = 's%i/lifted_nh_%s' % (scale, lifted_prefix)
    lifted_costs_key = 's%i/lifted_costs_%s' % (scale, lifted_prefix)
    ds = problem[nh_key]
    fu.log("reading lifted uvs")
    ds.n_threads = n_threads
    lifted_uvs = ds[:]

    fu.log("reading lifted costs")
    ds = problem[lifted_costs_key]
    ds.n_threads = n_threads
    lifted_costs = ds[:]

    # the output group
    out = problem['s%i/sub_results_lmc' % scale]

    # NOTE we use different sub-graph identifiers for multicut and lifted multicut
    # in order to run both in the same n5-container.
    # However, for scale level 0 the sub-graphs come from the GraphWorkflow and
    # are hence identical
    sub_graph_identifier = 'sub_graphs' if scale == 0 else 'sub_graphs_lmc'
    ds_nodes = problem['s%i/%s/nodes' % (scale, sub_graph_identifier)]
    blocking = nt.blocking([0, 0, 0], shape, list(block_shape))

    fu.log("start processsing %i blocks" % len(block_list))
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_solve_block_problem,
                           block_id, graph, uv_ids, ds_nodes,
                           costs, lifted_uvs, lifted_costs,
                           lifted_solver, solver, ignore_label,
                           blocking, out, time_limit)
                 for block_id in block_list]
        [t.result() for t in tasks]

    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    solve_lifted_subproblems(job_id, path)
