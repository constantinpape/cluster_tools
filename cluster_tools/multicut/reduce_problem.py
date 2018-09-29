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
# Multicut Tasks
#


class ReduceProblemBase(luigi.Task):
    """ ReduceProblem base class
    """

    task_name = 'reduce_problem'
    src_file = os.path.abspath(__file__)
    allow_retry = False

    # input volumes and graph
    costs_path = luigi.Parameter()
    costs_key = luigi.Parameter()
    graph_path = luigi.Parameter()
    graph_key = luigi.Parameter()
    output_path = luigi.Parameter()
    scale = luigi.IntParameter()
    #
    dependency = luigi.TaskParameter()

    def requires(self):
        return self.dependency

    @staticmethod
    def default_task_config():
        # we use this to get also get the common default config
        config = LocalTask.default_task_config()
        config.update({'accumulation_method': 'sum'})
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
                       'output_path': self.output_path, 'tmp_folder': self.tmp_folder,
                       'scale': self.scale, 'block_shape': block_shape})
        if roi_begin is not None:
            assert roi_end is not None
            config.update({'roi_begin': roi_begin,
                           'roi_end': roi_end})

        with vu.file_reader(self.graph_path, 'r') as f:
            shape = f.attrs['shape']

        factor = 2**self.scale
        block_shape = tuple(bs * factor for bs in block_shape)

        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)
        config.update({'n_jobs': n_jobs})

        # prime and run the job
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)

    # part of the luigi API
    def output(self):
        return luigi.LocalTarget(os.path.join(self.tmp_folder,
                                              self.task_name + '_s%i.log' % self.scale))


class ReduceProblemLocal(ReduceProblemBase, LocalTask):
    """ ReduceProblem on local machine
    """
    pass


class ReduceProblemSlurm(ReduceProblemBase, SlurmTask):
    """ ReduceProblem on slurm cluster
    """
    pass


class ReduceProblemLSF(ReduceProblemBase, LSFTask):
    """ ReduceProblem on lsf cluster
    """
    pass


#
# Implementation
#


def _merge_nodes(tmp_folder, scale, n_jobs, nodes, uv_ids, initial_node_labeling):
    n_edges = len(uv_ids)
    # load the cut-edge ids from the prev. jobs and make merge edge ids
    # TODO we could parallelize this
    cut_edge_ids = np.concatenate([np.load(os.path.join(tmp_folder, 'subproblem_results',
                                                        's%i_job%i.npy' % (scale, job_id)))
                                   for job_id in range(n_jobs)]).astype('uint64')
    # we only need to run unique if we have more than one job
    if n_jobs > 1:
        cut_edge_ids = np.unique(cut_edge_ids)

    assert len(cut_edge_ids) < n_edges, "%i = %i, does not reduce problem" % (len(cut_edge_ids),
                                                                              n_edges)

    merge_edges = np.ones(n_edges, dtype='bool')
    merge_edges[cut_edge_ids] = False

    # merge node pairs with ufd
    ufd = nufd.boost_ufd(nodes)
    ufd.merge(uv_ids[merge_edges])

    # get the node results and label them consecutively
    node_labeling = ufd.find(nodes)
    node_labeling, max_new_id, _ = relabelConsecutive(node_labeling, start_label=0, keep_zeros=False)
    # make sure that zero is still mapped to zero
    if node_labeling[0] != 0:
        # if it isn't, swap labels accordingly
        zero_label = node_labeling[0]
        to_relabel = node_labeling == 0
        node_labeling[node_labeling == zero_label] = 0
        node_labeling[to_relabel] = zero_laebl
    n_new_nodes = max_new_id + 1

    # get the labeling of initial nodes
    if initial_node_labeling is None:
        # if we don't have an initial node labeling, we are in the first scale.
        # here, the graph nodes might not be consecutive / not start at zero.
        # to keep the node labeling valid, we must make the labeling consecutive by inserting zeros

        # check if `nodes` are consecutive and start at zero
        node_max_id = int(nodes.max())
        if node_max_id + 1 != len(nodes):
            fu.log("nodes are not consecutve and/or don't start at zero")
            fu.log("inflating node labels accordingly")
            node_labeling = nt.inflateLabeling(nodes, node_labeling, node_max_id)

        new_initial_node_labeling = node_labeling
    else:
        # NOTE access like this is ok because all node labelings will be consecutive
        new_initial_node_labeling = node_labeling[initial_node_labeling]

    return n_new_nodes, node_labeling, new_initial_node_labeling


def _get_new_edges(uv_ids, node_labeling, costs, accumulation_method, n_threads):
    edge_mapping = nt.EdgeMapping(uv_ids, node_labeling, numberOfThreads=n_threads)
    new_uv_ids = edge_mapping.newUvIds()
    edge_labeling = edge_mapping.edgeMapping()
    new_costs = edge_mapping.mapEdgeValues(costs, accumulation_method,
                                           numberOfThreads=n_threads)
    assert new_uv_ids.max() == node_labeling.max(), "%i, %i" % (new_uv_ids.max(),
                                                                node_labeling.max())
    assert len(new_uv_ids) == len(new_costs)
    assert len(edge_labeling) == len(uv_ids)

    return new_uv_ids, edge_labeling, new_costs


def _serialize_new_problem(graph_path, graph_key,
                           n_new_nodes, new_uv_ids,
                           node_labeling, edge_labeling,
                           new_costs, new_initial_node_labeling,
                           shape, scale, initial_block_shape,
                           output_path, n_threads,
                           roi_begin, roi_end):

    next_scale = scale + 1
    f_out= z5py.File(output_path)
    g_out = f_out.require_group('s%i' % next_scale)
    g_out.require_group('sub_graphs')

    block_in_prefix = os.path.join(output_path, 's%i' % scale, 'sub_graphs', 'block_')
    block_out_prefix = os.path.join(output_path, 's%i' % next_scale, 'sub_graphs', 'block_')

    factor = 2**scale
    block_shape = [factor * bs for bs in initial_block_shape]

    new_factor = 2**(scale + 1)
    new_block_shape = [new_factor * bs for bs in initial_block_shape]

    # NOTE we do not need to serialize the sub-edges in the current implementation
    # of the blockwise multicut workflow, because we always load the full graph
    # in 'solve_subproblems'
    # serialize the new sub-graphs
    block_ids = vu.blocks_in_volume(shape, new_block_shape, roi_begin, roi_end)
    ndist.serializeMergedGraph(graphBlockPrefix=block_in_prefix,
                               shape=shape,
                               blockShape=block_shape,
                               newBlockShape=new_block_shape,
                               newBlockIds=block_ids,
                               nodeLabeling=node_labeling,
                               edgeLabeling=edge_labeling,
                               graphOutPrefix=block_out_prefix,
                               numberOfThreads=n_threads,
                               serializeEdges=False)

    # serialize the full graph for the next scale level
    n_new_edges = len(new_uv_ids)
    g_out.attrs['numberOfNodes'] = n_new_nodes
    g_out.attrs['numberOfEdges'] = n_new_edges

    with vu.file_reader(graph_path, 'r') as f:
        ignore_label = f[graph_key].attrs['ignoreLabel']
    g_out.attrs['ignoreLabel'] = ignore_label

    def _serialize(name, data, dtype='uint64'):
        ser_chunks = (min(data.shape[0], 262144), 2) if data.ndim == 2 else\
            (min(data.shape[0], 262144),)
        ds_ser = g_out.require_dataset(name, dtype=dtype, shape=data.shape,
                                       chunks=ser_chunks, compression='gzip')
        ds_ser.n_threads = n_threads
        ds_ser[:] = data

    # NOTE we don not need to serialize the nodes cause they are
    # consecutive anyways
    # _serialize('nodes', np.arange(n_new_nodes).astype('uint64'))
    # serialize the new graph, the node labeling and the new costs
    _serialize('edges', new_uv_ids)
    _serialize('node_labeling', new_initial_node_labeling)
    _serialize('costs', new_costs, dtype='float32')

    return n_new_edges


def reduce_problem(job_id, config_path):

    fu.log("start processing job %i" % job_id)
    fu.log("reading config from %s" % config_path)

    # get the config
    with open(config_path) as f:
        config = json.load(f)
    graph_path = config['graph_path']
    graph_key = config['graph_key']
    costs_path = config['costs_path']
    costs_key = config['costs_key']
    output_path = config['output_path']
    tmp_folder= config['tmp_folder']
    initial_block_shape = config['block_shape']
    scale = config['scale']
    n_jobs = config['n_jobs']
    accumulation_method = config.get('accumulation_method', 'sum')
    n_threads = config['threads_per_job']
    roi_begin = config.get('roi_begin', None)
    roi_end = config.get('roi_end', None)

    # get the number of nodes and uv-ids at this scale level
    # as well as the initial node labeling
    fu.log("read graph from %s, %s" % (graph_path, graph_key))
    with vu.file_reader(graph_path, 'r') as f:
        shape = f.attrs['shape']

        # load graph nodes and edges
        group = f[graph_key]

        # nodes
        # we only need to load the nodes for scale 0
        # otherwise, we already know that they are consecutive
        if scale == 0:
            ds = group['nodes']
            ds.n_threads = n_threads
            nodes = ds[:]
            n_nodes = len(nodes)
        else:
            n_nodes = group.attrs['numberOfNodes']
            nodes = np.arange(n_nodes, dtype='uint64')

        # edges
        ds = group['edges']
        ds.n_threads = n_threads
        uv_ids = ds[:]
        n_edges = len(uv_ids)

        # read initial node labeling
        if scale == 0:
            initial_node_labeling = None
        else:
            ds = group['node_labeling']
            ds.n_threads = n_threads
            initial_node_labeling = ds[:]

    fu.log("read costs from %s, %s" % (costs_path, costs_key))
    with vu.file_reader(costs_path) as f:
        ds = f[costs_key]
        ds.n_threads = n_threads
        costs = ds[:]
    assert len(costs) == n_edges, "%i, %i" (len(costs), n_edges)

    # get the new node assignment
    fu.log("merge nodes")
    n_new_nodes, node_labeling, new_initial_node_labeling = _merge_nodes(tmp_folder,
                                                                         scale,
                                                                         n_jobs,
                                                                         nodes,
                                                                         uv_ids,
                                                                         initial_node_labeling)
    # get the new edge assignment
    fu.log("get new edge ids")
    new_uv_ids, edge_labeling, new_costs = _get_new_edges(uv_ids, node_labeling,
                                                          costs, accumulation_method, n_threads)

    # serialize the input graph and costs for the next scale level
    fu.log("serialize new problem to %s/s%i" % (output_path, scale))
    n_new_edges = _serialize_new_problem(graph_path, graph_key,
                                         n_new_nodes, new_uv_ids,
                                         node_labeling, edge_labeling,
                                         new_costs, new_initial_node_labeling,
                                         shape, scale, initial_block_shape,
                                         output_path, n_threads,
                                         roi_begin, roi_end)

    fu.log("Reduced graph from %i to %i nodes; %i to %i edges." % (n_nodes, n_new_nodes,
                                                                   n_edges, n_new_edges))
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    reduce_problem(job_id, path)
