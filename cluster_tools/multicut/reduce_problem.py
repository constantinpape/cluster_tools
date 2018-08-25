#! /bin/python

import os
import sys
import json
from concurrent import futures

import luigi
import z5py
import nifty.tools as nt
# TODO boost ufd seems to be faster than nifty ufd,
# maybe should implement wrapper
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
                       'scale': self.scale, 'block_shape': block_shape})

        with vu.file_reader(self.graph_path, 'r') as f:
            shape = f.attrs['shape']

        block_list = vu.blocks_in_volume(shape, block_shape, roi_begin, roi_end)
        n_jobs = min(len(block_list), self.max_jobs)
        config.update({'n_jobs': n_jobs})

        # prime and run the job
        self.prepare_jobs(1, None, config)
        self.submit_jobs(1)

        # wait till jobs finish and check for job success
        self.wait_for_jobs()
        self.check_jobs(1)


class ReduceProblemLocal(ReduceProblemBase, LocalTask):
    """ ReduceProblem on local machine
    """
    pass


class ReduceProblemsSlurm(ReduceProblemBase, SlurmTask):
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


def _merge_nodes(tmp_folder, scale, n_jobs, n_nodes, uv_ids, initial_node_labeling):
    n_edges = len(uv_ids)
    # load the cut-edge ids from the prev. jobs and make merge edge ids
    # TODO we could parallelize this
    cut_edge_ids = np.concatenate([np.load(os.path.join(tmp_folder,
                                                        '1_output_s%i_%i.npy' % (scale, job_id)))
                                   for job_id in range(n_jobs)])
    cut_edge_ids = np.unique(cut_edge_ids).astype('uint64')

    # print("Number of cut edges:", len(cut_edge_ids))
    # print("                   /", n_edges)
    assert len(cut_edge_ids) < n_edges, "%i = %i, does not reduce problem" % (len(cut_edge_ids), n_edges)

    merge_edges = np.ones(n_edges, dtype='bool')
    merge_edges[cut_edge_ids] = False

    # TODO make sure that zero stayes mapped to zero

    # we don't have edges to zero any more, so do't need to do this
    # additionally, we make sure that all edges are cut
    # ignore_edges = (uv_ids == 0).any(axis=1)
    # merge_edges[ignore_edges] = False

    # merge node pairs with ufd
    ufd = nufd.ufd(n_nodes)
    merge_pairs = uv_ids[merge_edges]
    ufd.merge(merge_pairs)

    # get the node results and label them consecutively
    node_labeling = ufd.elementLabeling()
    node_labeling, max_new_id, _ = relabelConsecutive(node_labeling)
    n_new_nodes = max_new_id + 1

    # get the labeling of initial nodes
    if initial_node_labeling is None:
        new_initial_node_labeling = node_labeling
    else:
        # should this ever become a bottleneck, we can parallelize this in nifty
        # but for now this would really be premature optimization
        new_initial_node_labeling = node_labeling[initial_node_labeling]

    return n_new_nodes, node_labeling, new_initial_node_labeling


def _get_new_edges(uv_ids, node_labeling, costs, accumulation_method, n_threads):
    edge_mapping = nt.EdgeMapping(uv_ids, node_labeling, numberOfThreads=n_threads)
    new_uv_ids = edge_mapping.newUvIds()
    edge_labeling = edge_mapping.edgeMapping()
    new_costs = edge_mapping.mapEdgeValues(costs, accumulation_method, numberOfThreads=n_threads)
    assert len(new_uv_ids) == len(new_costs)
    assert len(edge_labeling) == len(uv_ids)
    return new_uv_ids, edge_labeling, new_costs


def _serialize_new_problem(graph_path, n_new_nodes, new_uv_ids,
                           node_labeling, edge_labeling,
                           new_costs, new_initial_node_labeling,
                           shape, scale, initial_block_shape,
                           tmp_folder, n_threads):

    next_scale = scale + 1
    merged_graph_path = os.path.join(tmp_folder, 'merged_graph.n5')
    f_graph = z5py.File(merged_graph_path)
    g_out = f_graph.require_group('s%i' % next_scale)
    g_out.require_group('sub_graphs')

    # TODO this should be handled by symlinks
    if scale == 0:
        block_in_prefix = os.path.join(graph_path, 'sub_graphs', 's%i' % scale, 'block_')
    else:
        block_in_prefix = os.path.join(tmp_folder, 'merged_graph.n5', 's%i' % scale, 'sub_graphs', 'block_')

    block_out_prefix = os.path.join(tmp_folder, 'merged_graph.n5', 's%i' % next_scale, 'sub_graphs', 'block_')

    factor = 2**scale
    block_shape = [factor * bs for bs in initial_block_shape]

    new_factor = 2**(scale + 1)
    new_block_shape = [new_factor * bs for bs in initial_block_shape]

    ndist.serializeMergedGraph(block_in_prefix, shape,
                               block_shape, new_block_shape,
                               n_new_nodes,
                               node_labeling, edge_labeling,
                               block_out_prefix, n_threads)

    # serialize the full graph for the next scale level
    n_new_edges = len(new_uv_ids)
    g_out.attrs['numberOfNodes'] = n_new_nodes
    g_out.attrs['numberOfEdges'] = n_new_edges

    shape_edges = (n_new_edges, 2)
    ds_edges = g_out.create_dataset('edges', dtype='uint64', shape=shape_edges, chunks=shape_edges)
    ds_edges[:] = new_uv_ids

    nodes = np.unique(new_uv_ids)
    shape_nodes = (len(nodes),)
    ds_nodes = g_out.create_dataset('nodes', dtype='uint64', shape=shape_nodes, chunks=shape_nodes)
    ds_nodes[:] = nodes

    # serialize the node labeling
    shape_node_labeling = (len(new_initial_node_labeling),)
    ds_node_labeling = g_out.create_dataset('nodeLabeling', dtype='uint64', shape=shape_node_labeling,
                                            chunks=shape_node_labeling)
    ds_node_labeling[:] = new_initial_node_labeling

    # serialize the new costs
    shape_costs = (n_new_edges,)
    ds_costs = g_out.require_dataset('costs', dtype='float32',
                                     shape=shape_costs, chunks=shape_costs)
    ds_costs[:] = new_costs

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
    block_shape = config['block_shape']
    scale = config['scale']
    n_jobs = config['n_jobs']
    accumulation_method = config.get('accumulation_method', 'sum')
    n_threads = config['threads_per_job']

    # get the number of nodes and uv-ids at this scale level
    # as well as the initial node labeling
    with vu.file_reader(graph_path, 'r') as f:
        shape = f.attrs['shape']
        group = f[graph_key]
        n_nodes = group.attrs['numberOfNodes']
        ds = group['edges']
        ds.n_threads = n_threads
        uv_ids = ds[:]

        if scale == 0:
            initial_node_labeling = None
        else:
            ds = group['nodeLabeling']
            ds.n_threads = n_threads
            initial_node_labeling = ds[:]

    n_edges = len(uv_ids)
    with vu.file_reader(costs_path) as f:
        ds = f[costs_key]
        ds.n_threads = n_threads
        costs = ds[:]
    assert len(costs) == n_edges, "%i, %i" (len(costs), n_edges)

    # get the new node assignment
    n_new_nodes, node_labeling, new_initial_node_labeling = _merge_nodes(tmp_folder,
                                                                         scale,
                                                                         n_jobs,
                                                                         n_nodes,
                                                                         uv_ids,
                                                                         initial_node_labeling)
    # get the new edge assignment
    new_uv_ids, edge_labeling, new_costs = _get_new_edges(uv_ids, node_labeling,
                                                          costs, accumulation_method, n_threads)

    # serialize the input graph and costs for the next scale level
    n_new_edges = _serialize_new_problem(graph_path, n_new_nodes, new_uv_ids,
                                         node_labeling, edge_labeling,
                                         new_costs, new_initial_node_labeling,
                                         shape, scale, initial_block_shape,
                                         tmp_folder, n_threads)

    fu.log("Reduced graph from %i to %i nodes; %i to %i edges." % (n_nodes, n_new_nodes,
                                                                   n_edges, n_new_edges))
    fu.log_job_success(job_id)


if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path), path
    job_id = int(os.path.split(path)[1].split('.')[0].split('_')[-1])
    reduce_problem(job_id, path)
