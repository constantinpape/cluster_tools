#! /usr/bin/python

import time
import os
import argparse
import numpy as np

from vigra.analysis import relabelConsecutive
import z5py
# we don't want to pull in big packges we don't need
import nifty.tools as nt
import nifty.ufd as nufd
import nifty.distributed as ndist


def merge_nodes(tmp_folder, scale, n_jobs, n_nodes, uv_ids, initial_node_labeling):
    n_edges = len(uv_ids)
    # load the cut-edge ids from the prev. jobs and make merge edge ids
    # TODO we could parallelize this
    cut_edge_ids = np.concatenate([np.load(os.path.join(tmp_folder,
                                                        '1_output_s%i_%i.npy' % (scale, job_id)))
                                   for job_id in range(n_jobs)])
    cut_edge_ids = np.unique(cut_edge_ids)

    # print("Number of cut edges:", len(cut_edge_ids))
    # print("                   /", n_edges)
    assert len(cut_edge_ids) < n_edges, "%i = %i, does not reduce problem" % (len(cut_edge_ids), n_edges)

    merge_edges = np.ones(n_edges, dtype='bool')
    merge_edges[cut_edge_ids] = False

    # TODO make sure that zero stayes mapped to zero
    # additionally, we make sure that all edges are cut
    ignore_edges = (uv_ids == 0).any(axis=1)
    merge_edges[ignore_edges] = False

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


def get_new_edges(uv_ids, node_labeling, costs, cost_accumulation, n_threads):
    edge_mapping = nt.EdgeMapping(uv_ids, node_labeling, numberOfThreads=n_threads)
    new_uv_ids = edge_mapping.newUvIds()
    edge_labeling = edge_mapping.edgeMapping()
    new_costs = edge_mapping.mapEdgeValues(costs, cost_accumulation, numberOfThreads=n_threads)
    assert len(new_uv_ids) == len(new_costs)
    assert len(edge_labeling) == len(uv_ids)
    return new_uv_ids, edge_labeling, new_costs


def serialize_new_problem(graph_path, n_new_nodes, new_uv_ids,
                          node_labeling, edge_labeling,
                          new_costs, new_initial_node_labeling,
                          shape, scale, initial_block_shape,
                          tmp_folder, n_threads):

    next_scale = scale + 1
    merged_graph_path = os.path.join(tmp_folder, 'merged_graph.n5')
    f_graph = z5py.File(merged_graph_path, use_zarr_format=False)
    g_out = f_graph.create_group('s%i' % next_scale)
    g_out.create_group('sub_graphs')

    # TODO this should be handled by symlinks
    if scale == 0:
        block_in_prefix = os.path.join(graph_path, 'sub_graphs', 's%i' % scale, 'block_')
    else:
        block_in_prefix = os.path.join(tmp_folder, 'merged_graph.n5', 's%i' % scale, 'block_')

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
    if 'costs' not in g_out:
        ds_costs = g_out.create_dataset('costs', dtype='float32', shape=shape_costs, chunks=shape_costs)
    else:
        ds_costs = g_out['costs']
    ds_costs[:] = new_costs

    return n_new_edges


def multicut_step1(graph_path, scale,
                   tmp_folder, n_jobs,
                   initial_block_shape, n_threads,
                   cost_accumulation="sum"):
    t0 = time.time()
    # get the number of nodes and uv-ids at this scale level
    # as well as the initial node labeling
    shape = z5py.File(graph_path).attrs['shape']
    if scale == 0:
        f_graph = z5py.File(os.path.join(graph_path, 'graph'), use_zarr_format=False)
        initial_node_labeling = None
        n_nodes = f_graph.attrs['numberOfNodes']
        uv_ids = f_graph['edges'][:]
    else:
        f_problem = z5py.File(os.path.join(tmp_folder, 'merged_graph.n5/s%i' % scale))
        n_nodes = f_problem.attrs['numberOfNodes']
        uv_ids = f_problem['uvIds'][:]
        initial_node_labeling = f_problem['nodeLabeling'][:]
    n_edges = len(uv_ids)

    # get the costs
    costs = z5py.File(os.path.join(tmp_folder, 'merged_graph.n5/s%i' % scale),
                      use_zarr_format=False)['costs'][:]
    assert len(costs) == n_edges, "%i, %i" (len(costs), n_edges)

    # get the new node assignment
    n_new_nodes, node_labeling, new_initial_node_labeling = merge_nodes(tmp_folder,
                                                                        scale,
                                                                        n_jobs,
                                                                        n_nodes,
                                                                        uv_ids,
                                                                        initial_node_labeling)
    # get the new edge assignment
    new_uv_ids, edge_labeling, new_costs = get_new_edges(uv_ids, node_labeling,
                                                         costs, cost_accumulation, n_threads)

    # serialize the input graph and costs for the next scale level
    n_new_edges = serialize_new_problem(graph_path, n_new_nodes, new_uv_ids,
                                        node_labeling, edge_labeling,
                                        new_costs, new_initial_node_labeling,
                                        shape, scale, initial_block_shape,
                                        tmp_folder, n_threads)

    print("Success")
    print("In %f s" % (time.time() - t0,))
    print("Reduced graph from", n_nodes, "to", n_new_nodes, "nodes;",
          n_edges, "to", n_new_edges, "edges.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path", type=str)
    parser.add_argument("scale", type=int)
    parser.add_argument("--tmp_folder", type=str)
    parser.add_argument("--n_jobs", type=int)
    parser.add_argument("--initial_block_shape", type=int, nargs=3)
    parser.add_argument("--n_threads", type=int)
    parser.add_argument("--cost_accumulation", type=str)
    args = parser.parse_args()

    multicut_step1(args.graph_path,
                   args.scale, args.tmp_folder,
                   args.n_jobs, args.initial_block_shape,
                   args.n_threads, args.cost_accumulation)
