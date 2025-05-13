#! /bin/python

import os
import sys
import json
from concurrent import futures

import numpy as np
import luigi
import z5py

import nifty.tools as nt
import nifty.ufd as nufd
import nifty.distributed as ndist
from vigra.analysis import relabelConsecutive

import cluster_tools.utils.volume_utils as vu
import cluster_tools.utils.function_utils as fu
from cluster_tools.cluster_tasks import SlurmTask, LocalTask, LSFTask

#
# Lifted Multicut Tasks
#


class ReduceLiftedProblemBase(luigi.Task):
    """ ReduceLiftedProblem base class
    """

    task_name = "reduce_lifted_problem"
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input volumes and graph
    problem_path = luigi.Parameter()
    scale = luigi.IntParameter()
    lifted_prefix = luigi.Parameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({"accumulation_method": "sum"})
        return config

    # TODO log reduction of lifted edges
    def _log_reduction(self):
        key1 = "s0/graph" if self.scale == 0 else "s%i/graph_lmc" % self.scale
        key2 = "s%i/graph_lmc" % (self.scale + 1,)
        with vu.file_reader(self.problem_path, "r") as f:
            n_nodes = f[key1].attrs["numberOfNodes"]
            n_edges = f[key1].attrs["numberOfEdges"]
            n_new_nodes = f[key2].attrs["numberOfNodes"]
            n_new_edges = f[key2].attrs["numberOfEdges"]
        self._write_log("Reduced graph from %i to %i nodes; %i to %i edges." % (n_nodes, n_new_nodes,
                                                                                n_edges, n_new_edges))

    def run_impl(self):
        # get the global config and init configs
        shebang, block_shape, roi_begin, roi_end = self.global_config_values()
        self.init(shebang)

        # load the task config
        config = self.get_task_config()

        # update the config with input and graph paths and keys
        # as well as block shape
        config.update({"problem_path": self.problem_path, "scale": self.scale,
                       "block_shape": block_shape, "lifted_prefix": self.lifted_prefix})
        if roi_begin is not None:
            assert roi_end is not None
            config.update({"roi_begin": roi_begin,
                           "roi_end": roi_end})

        with vu.file_reader(self.problem_path, "r") as f:
            shape = f["s0/graph"].attrs["shape"]

        factor = 2**self.scale
        block_shape = tuple(bs * factor for bs in block_shape)

        # prime and run the job
        prefix = "s%i" % self.scale
        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        self.prepare_jobs(1, block_list, config, prefix)
        self.submit_jobs(1, prefix)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1, prefix)

        # log the problem reduction
        self._log_reduction()

    # part of the luigi API
    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + "_s%i.log" % self.scale))


class ReduceLiftedProblemLocal(ReduceLiftedProblemBase, LocalTask):
    """ ReduceLiftedProblem on local machine
    """
    pass


class ReduceLiftedProblemSlurm(ReduceLiftedProblemBase, SlurmTask):
    """ ReduceLiftedProblem on slurm cluster
    """
    pass


class ReduceLiftedProblemLSF(ReduceLiftedProblemBase, LSFTask):
    """ ReduceLiftedProblem on lsf cluster
    """
    pass


#
# Implementation
#

def _load_cut_edges(problem_path, scale, blocking,
                    block_list, n_threads):
    key = "s%i/sub_results_lmc/cut_edge_ids" % scale
    ds = z5py.File(problem_path)[key]

    def load_block_res(block_id):
        block = blocking.getBlock(block_id)
        chunk_id = tuple(beg // sh for beg, sh
                         in zip(block.begin, blocking.blockShape))
        return ds.read_chunk(chunk_id)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(load_block_res, block_id)
                 for block_id in block_list]
        cut_edge_ids = [t.result() for t in tasks]
        cut_edge_ids = np.concatenate([ids for ids in cut_edge_ids
                                       if ids is not None])

    return np.unique(cut_edge_ids)


def _merge_nodes(problem_path, scale, blocking,
                 block_list, nodes, uv_ids,
                 initial_node_labeling, n_threads):
    # load the cut edge ids
    n_edges = len(uv_ids)
    cut_edge_ids = _load_cut_edges(problem_path, scale, blocking, block_list, n_threads)
    assert len(cut_edge_ids) < n_edges, "%i = %i, does not reduce problem" % (len(cut_edge_ids), n_edges)

    merge_edges = np.ones(n_edges, dtype="bool")
    merge_edges[cut_edge_ids] = False
    fu.log("merging %i / %i edges" % (np.sum(merge_edges), n_edges))

    # merge node pairs with ufd
    n_nodes = int(nodes.max()) + 1
    ufd = nufd.ufd(n_nodes)
    ufd.merge(uv_ids[merge_edges])

    # get the node results and label them consecutively
    node_labeling = ufd.find(nodes)
    node_labeling, max_new_id, _ = relabelConsecutive(node_labeling, start_label=0, keep_zeros=False)
    assert node_labeling[0] == 0
    # FIXME this looks fishy, redo !!!
    # # make sure that zero is still mapped to zero
    # if node_labeling[0] != 0:
    #     # if it isn"t, swap labels accordingly
    #     zero_label = node_labeling[0]
    #     to_relabel = node_labeling == 0
    #     node_labeling[node_labeling == zero_label] = 0
    #     node_labeling[to_relabel] = zero_laebl
    n_new_nodes = max_new_id + 1
    fu.log("have %i nodes in new node labeling" % n_new_nodes)

    # get the labeling of initial nodes
    if initial_node_labeling is None:
        # if we don"t have an initial node labeling, we are in the first scale.
        # here, the graph nodes might not be consecutive / not start at zero.
        # to keep the node labeling valid, we must make the labeling consecutive by inserting zeros
        fu.log("don't have an initial node labeling")

        # check if `nodes` are consecutive and start at zero
        node_max_id = int(nodes.max())
        if node_max_id + 1 != len(nodes):
            fu.log("nodes are not consecutve and/or don't start at zero")
            fu.log("inflating node labels accordingly")
            node_labeling = nt.inflateLabeling(nodes, node_labeling, node_max_id)

        new_initial_node_labeling = node_labeling
    else:
        fu.log("mapping new node labeling to labeling of inital (= scale 0) nodes")
        # NOTE access like this is ok because all node labelings will be consecutive
        new_initial_node_labeling = node_labeling[initial_node_labeling]
        assert len(new_initial_node_labeling) == len(initial_node_labeling)

    return n_new_nodes, node_labeling, new_initial_node_labeling


def _get_new_edges(uv_ids, node_labeling, costs, accumulation_method, n_threads):
    edge_mapping = nt.EdgeMapping(uv_ids, node_labeling, numberOfThreads=n_threads)
    new_uv_ids = edge_mapping.newUvIds()
    edge_labeling = edge_mapping.edgeMapping()
    new_costs = edge_mapping.mapEdgeValues(costs, accumulation_method,
                                           numberOfThreads=n_threads)
    assert new_uv_ids.max() <= node_labeling.max(), "%i, %i" % (new_uv_ids.max(),
                                                                node_labeling.max())
    assert len(new_uv_ids) == len(new_costs)
    assert len(edge_labeling) == len(uv_ids)

    return new_uv_ids, edge_labeling, new_costs


def _serialize_new_problem(problem_path,
                           n_new_nodes, new_uv_ids,
                           node_labeling, edge_labeling,
                           new_costs, new_initial_node_labeling,
                           new_lifted_uvs, new_lifted_costs,
                           shape, scale, initial_block_shape,
                           n_threads, roi_begin, roi_end,
                           lifted_prefix):

    assert len(new_costs) == len(new_uv_ids)
    assert len(new_lifted_uvs) == len(new_lifted_costs)
    next_scale = scale + 1
    f_out = z5py.File(problem_path)
    g_out = f_out.require_group("s%i" % next_scale)

    # NOTE we use different sub-graph identifiers for multicut and lifted multicut
    # in order to run both in the same n5-container.
    # However, for scale level 0 the sub-graphs come from the GraphWorkflow and
    # are hence identical
    sub_graph_identifier = "sub_graphs" if scale == 0 else "sub_graphs_lmc"
    g_out.require_group(sub_graph_identifier)

    subgraph_in_key = "s%i/%s" % (scale, sub_graph_identifier)
    subgraph_out_key = "s%i/sub_graphs_lmc" % next_scale

    factor = 2**scale
    block_shape = [factor * bs for bs in initial_block_shape]

    new_factor = 2**(scale + 1)
    new_block_shape = [new_factor * bs for bs in initial_block_shape]

    # NOTE we do not need to serialize the sub-edges in the current implementation
    # of the blockwise multicut workflow, because we always load the full graph
    # in "solve_subproblems"

    # serialize the new sub-graphs
    block_ids = vu.blocks_in_volume(shape, new_block_shape, roi_begin, roi_end)
    ndist.serializeMergedGraph(graphPath=problem_path,
                               graphBlockPrefix=subgraph_in_key,
                               shape=shape,
                               blockShape=block_shape,
                               newBlockShape=new_block_shape,
                               newBlockIds=block_ids,
                               nodeLabeling=node_labeling,
                               edgeLabeling=edge_labeling,
                               outPath=problem_path,
                               graphOutPrefix=subgraph_out_key,
                               numberOfThreads=n_threads,
                               serializeEdges=False)

    # serialize the multicut problem for the next scale level

    graph_key = "s%i/graph_lmc" % scale if scale > 0 else "s0/graph"
    with vu.file_reader(problem_path, "r") as f:
        ignore_label = f[graph_key].attrs["ignore_label"]

    n_new_edges = len(new_uv_ids)
    graph_out = g_out.require_group("graph_lmc")
    graph_out.attrs["ignore_label"] = ignore_label
    graph_out.attrs["numberOfNodes"] = n_new_nodes
    graph_out.attrs["numberOfEdges"] = n_new_edges
    graph_out.attrs["shape"] = shape

    def _serialize(out_group, name, data, dtype="uint64"):
        ser_chunks = (min(data.shape[0], 262144), 2) if data.ndim == 2 else\
            (min(data.shape[0], 262144),)
        ds_ser = out_group.require_dataset(name, dtype=dtype, shape=data.shape,
                                           chunks=ser_chunks, compression="gzip")
        ds_ser.n_threads = n_threads
        ds_ser[:] = data

    # NOTE we don not need to serialize the nodes cause they are
    # consecutive anyway
    # _serialize("nodes", np.arange(n_new_nodes).astype("uint64"))

    # serialize the new graph, the node labeling and the new costs
    _serialize(graph_out, "edges", new_uv_ids)
    _serialize(g_out, "node_labeling_lmc", new_initial_node_labeling)
    _serialize(g_out, "costs_lmc", new_costs, dtype="float32")
    # serialize lifted uvs and costs
    _serialize(g_out, "lifted_nh_%s" % lifted_prefix, new_lifted_uvs)
    _serialize(g_out, "lifted_costs_%s" % lifted_prefix, new_lifted_costs, dtype="float32")

    return n_new_edges


def reduce_lifted_problem(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    problem_path = config["problem_path"]
    initial_block_shape = config["block_shape"]
    scale = config["scale"]
    block_list = config["block_list"]
    lifted_prefix = config["lifted_prefix"]

    accumulation_method = config.get("accumulation_method", "sum")
    n_threads = config["threads_per_job"]
    roi_begin = config.get("roi_begin", None)
    roi_end = config.get("roi_end", None)

    # get the number of nodes and uv-ids at this scale level
    # as well as the initial node labeling
    fu.log("read problem from %s" % problem_path)

    # NOTE we use different graph identifiers for multicut and lifted multicut
    # in order to run both in the same n5-container.
    # However, for scale level 0 the graph comes from the GraphWorkflow and
    # hence the identifier is identical
    graph_key = "s%i/graph_lmc" % scale if scale > 0 else "s0/graph"
    with vu.file_reader(problem_path, "r") as f:
        shape = f[graph_key].attrs["shape"]

        # load graph nodes and edges
        group = f[graph_key]

        # nodes
        # we only need to load the nodes for scale 0
        # otherwise, we already know that they are consecutive
        if scale == 0:
            ds = group["nodes"]
            ds.n_threads = n_threads
            nodes = ds[:]
            n_nodes = len(nodes)
        else:
            n_nodes = group.attrs["numberOfNodes"]
            nodes = np.arange(n_nodes, dtype="uint64")

        # edges
        ds = group["edges"]
        ds.n_threads = n_threads
        uv_ids = ds[:]
        n_edges = len(uv_ids)

        # costs
        # NOTE we use different cost identifiers for multicut and lifted multicut
        # in order to run both in the same n5-container.
        # However, for scale level 0 the costs come from the CostsWorkflow and
        # hence the identifier is identical
        costs_key = "s%i/costs_lmc" % scale if scale > 0 else "s0/costs"
        ds = f[costs_key]
        ds.n_threads = n_threads
        costs = ds[:]
        assert len(costs) == n_edges, f"{len(costs)}, {n_edges}"

        # lifted edges
        nh_key = "s%i/lifted_nh_%s" % (scale, lifted_prefix)
        ds = f[nh_key]
        ds.n_threads = n_threads
        lifted_uvs = ds[:]

        lifted_costs_key = "s%i/lifted_costs_%s" % (scale, lifted_prefix)
        ds = f[lifted_costs_key]
        ds.n_threads = n_threads
        lifted_costs = ds[:]

        # read initial node labeling
        if scale == 0:
            initial_node_labeling = None
        else:
            ds = f["s%i/node_labeling_lmc" % scale]
            ds.n_threads = n_threads
            initial_node_labeling = ds[:]

    block_shape = [bsh * 2**scale for bsh in initial_block_shape]
    blocking = nt.blocking([0, 0, 0], shape, block_shape)

    # get the new node assignment
    fu.log("merge nodes")
    n_new_nodes, node_labeling, new_initial_node_labeling = _merge_nodes(problem_path, scale, blocking,
                                                                         block_list, nodes, uv_ids,
                                                                         initial_node_labeling, n_threads)
    # get the new edge assignment
    fu.log("get new edge ids")
    new_uv_ids, edge_labeling, new_costs = _get_new_edges(uv_ids, node_labeling,
                                                          costs, accumulation_method, n_threads)

    # get the new lifted edge assignment
    fu.log("get new lifted edge ids")
    new_lifted_uvs, _, new_lifted_costs = _get_new_edges(lifted_uvs, node_labeling,
                                                         lifted_costs, accumulation_method, n_threads)

    # serialize the input graph and costs for the next scale level
    fu.log("serialize new problem to %s/s%i" % (problem_path, scale + 1))
    n_new_edges = _serialize_new_problem(problem_path,
                                         n_new_nodes, new_uv_ids,
                                         node_labeling, edge_labeling,
                                         new_costs, new_initial_node_labeling,
                                         new_lifted_uvs, new_lifted_costs,
                                         shape, scale, initial_block_shape,
                                         n_threads, roi_begin, roi_end,
                                         lifted_prefix)

    fu.log("Reduced graph from %i to %i nodes; %i to %i edges; %i to %i lifted edges." % (n_nodes, n_new_nodes,
                                                                                          n_edges, n_new_edges,
                                                                                          len(lifted_uvs),
                                                                                          len(new_lifted_uvs)))
    fu.log_job_success(job_id)


if __name__ == "__main__":
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split(".")[0].split("_")[-1])
    reduce_lifted_problem(job_id, path)
